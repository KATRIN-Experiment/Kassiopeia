#include "KROOTWindow.h"

#include "KROOTPad.h"
#include "KROOTPainter.h"
#include "KUtilityMessage.h"
#include "KGlobals.hh"

#ifdef KASPER_USE_BOOST
//#include "KPathUtils.h"
//using katrin::KPathUtils;
#endif

#include <TQObject.h>
#include <TStyle.h>
#include <cmath>
#include <limits>

using namespace std;

namespace katrin
{

KROOTWindow::KROOTWindow() :
    fPainters(),
    fPads(),
    fFrame(nullptr),
    fApplication(nullptr),
    fCanvas(nullptr),
    fCanvasWidth(1000),
    fCanvasHeight(600),
    fActive(true),
    fWriteEnabled(false),
    fPath(""),
    fXMin(0.),
    fXMax(0.),
    fYMin(0.),
    fYMax(0.)
{}

KROOTWindow::~KROOTWindow()
{
    return;
}

void KROOTWindow::Render()
{
    utilmsg(eInfo) << "KROOTWindow starts to render!" << eom;

    //        gStyle->SetPadBottomMargin(0.1);
    //          gStyle->SetPadRightMargin(0.1);
    //          gStyle->SetPadLeftMargin(0.1);
    //          gStyle->SetPadTopMargin(0.1);
    gStyle->SetTitleX(0.5);
    gStyle->SetTitleAlign(23);
    gStyle->SetTitleSize(0.08, "t");

    if (!KGlobals::GetInstance().IsBatchMode())
    {
        if (gApplication) {
            fApplication = gApplication;
        }
        else {
            fApplication = new TApplication("My ROOT Application", nullptr, nullptr);
        }

        TQObject::Connect("TCanvas", "Closed()", "TApplication", fApplication, "Terminate()");
        TQObject::Connect("TPad", "Closed()", "TApplication", fApplication, "Terminate()");
    }

    fCanvas = new TCanvas(GetName().c_str(), GetName().c_str(), 10, 10, fCanvasWidth, fCanvasHeight);

    double tXMin(std::numeric_limits<double>::max());
    double tXMax(-1.0 * std::numeric_limits<double>::max());
    double tYMin(std::numeric_limits<double>::max());
    double tYMax(-1.0 * std::numeric_limits<double>::max());

    /* render painters */
    PainterIt tIt;
    if (fPainters.size() > 0) {
        for (tIt = fPainters.begin(); tIt != fPainters.end(); tIt++) {
            (*tIt)->Render();
            double tLocalXMin = (*tIt)->GetXMin();
            if (tLocalXMin < tXMin)
                tXMin = tLocalXMin;
            double tLocalXMax = (*tIt)->GetXMax();
            if (tLocalXMax > tXMax)
                tXMax = tLocalXMax;
            double tLocalYMin = (*tIt)->GetYMin();
            if (tLocalYMin < tYMin)
                tYMin = tLocalYMin;
            double tLocalYMax = (*tIt)->GetYMax();
            if (tLocalYMax > tYMax)
                tYMax = tLocalYMax;
        }

        if (fXMin < fXMax) {
            tXMin = fXMin;
            tXMax = fXMax;
        }
        if (fYMin < fYMax) {
            tYMin = fYMin;
            tYMax = fYMax;
        }

        utilmsg_debug("XMin: " << tXMin << eom);
        utilmsg_debug("XMax: " << tXMax << eom);
        utilmsg_debug("YMin: " << tYMin << eom);
        utilmsg_debug("YMax: " << tYMax << eom);

        tXMin = tXMin - (tXMax - tXMin) / 20.;
        tXMax = tXMax + (tXMax - tXMin) / 20.;
        tYMin = tYMin - (tYMax - tYMin) / 20.;
        tYMax = tYMax + (tYMax - tYMin) / 20.;

        if (tXMin == tXMax) {
            tXMin = tXMin - tXMin / 20.;
            tXMax = tXMax + tXMax / 20.;
        }

        if (tYMin == tYMax) {
            tYMin = tYMin - tYMin / 20.;
            tYMax = tYMax + tYMax / 20.;
        }

        Int_t tNBins = 1000;
        fFrame = new TH2F(GetName().c_str(), "", tNBins, tXMin, tXMax, tNBins, tYMin, tYMax);
        fFrame->SetStats(false);

        //take axis label of last painter
        KROOTPainter* tLastPainter = fPainters.at(fPainters.size() - 1);
        if (tLastPainter) {
            fFrame->GetXaxis()->SetTitle(tLastPainter->GetXAxisLabel().c_str());
            fFrame->GetYaxis()->SetTitle(tLastPainter->GetYAxisLabel().c_str());
        }
        fFrame->GetXaxis()->SetTitleSize(0.05);
        fFrame->GetXaxis()->SetTitleOffset(1.0);
        fFrame->GetXaxis()->SetLabelSize(0.05);
        fFrame->GetYaxis()->SetTitleSize(0.05);
        fFrame->GetYaxis()->SetTitleOffset(1.0);
        fFrame->GetYaxis()->SetLabelSize(0.05);
    }

    //render pads
    PadIt tPadIt;
    for (tPadIt = fPads.begin(); tPadIt != fPads.end(); tPadIt++) {
        (*tPadIt)->Render();
    }

    utilmsg(eInfo) << "KROOTWindow finished to render!" << eom;

    return;
}

void KROOTWindow::Display()
{
    if (KGlobals::GetInstance().IsBatchMode()) {
        utilmsg(eWarning) << "KROOTWindow display disabled in batch mode"
                        << eom;
        return;
    }

    utilmsg(eInfo) << "KROOTWindow starts to display!" << eom;

    fCanvas->cd();
    if (fFrame) {
        fFrame->Draw("axis");
    }

    /* display painters */
    PainterIt tIt;
    for (tIt = fPainters.begin(); tIt != fPainters.end(); tIt++) {
        (*tIt)->Display();
    }

    /* display pads */
    PadIt tPadIt;
    for (tPadIt = fPads.begin(); tPadIt != fPads.end(); tPadIt++) {
        fCanvas->cd();
        (*tPadIt)->Display();
    }

    utilmsg(eInfo) << "KROOTWindow finished to display!" << eom;
    return;
}

void KROOTWindow::Write()
{
    utilmsg(eInfo) << "KROOTWindow starts to write!" << eom;

    if (fWriteEnabled) {
        string tOutputStringRoot;
        string tOutputStringPNG;
        if (fPath.empty()) {
            tOutputStringRoot = OUTPUT_DEFAULT_DIR + string("/") + GetName() + string(".root");
            tOutputStringPNG = OUTPUT_DEFAULT_DIR + string("/") + GetName() + string(".png");
        }
        else {
#ifdef KASPER_USE_BOOST
//                KPathUtils::MakeDirectory( fPath );
#endif

            tOutputStringRoot = fPath + string("/") + GetName() + string(".root");
            tOutputStringPNG = fPath + string("/") + GetName() + string(".png");
        }
        fCanvas->SaveAs(tOutputStringRoot.c_str());
        fCanvas->SaveAs(tOutputStringPNG.c_str());
    }

    /* write painters */
    PainterIt tIt;
    for (tIt = fPainters.begin(); tIt != fPainters.end(); tIt++) {
        (*tIt)->Write();
    }

    /* write pads */
    PadIt tPadIt;
    for (tPadIt = fPads.begin(); tPadIt != fPads.end(); tPadIt++) {
        (*tPadIt)->Write();
    }

    if (fActive && fApplication) {
        if (!fApplication->IsRunning()) {
            utilmsg(eInfo) << "KROOTWindow starting the TApplication!" << eom;
            fApplication->Run(true);
        }
    }

    utilmsg(eInfo) << "KROOTWindow finished to write!" << eom;
    return;
}

void KROOTWindow::AddPainter(KPainter* aPainter)
{
    auto* tPainter = dynamic_cast<KROOTPainter*>(aPainter);
    if (tPainter != nullptr) {
        fPainters.push_back(tPainter);
        tPainter->SetWindow(this);
        return;
    }
    utilmsg(eError) << "cannot add non-root painter <" << aPainter->GetName() << "> to root window <" << GetName()
                    << ">" << eom;
    return;
}
void KROOTWindow::RemovePainter(KPainter* aPainter)
{
    auto* tPainter = dynamic_cast<KROOTPainter*>(aPainter);
    if (tPainter != nullptr) {
        PainterIt tIt;
        for (tIt = fPainters.begin(); tIt != fPainters.end(); tIt++) {
            if ((*tIt) == tPainter) {
                fPainters.erase(tIt);
                tPainter->ClearWindow(this);
                return;
            }
        }
        utilmsg(eError) << "cannot remove root painter <" << tPainter->GetName() << "> from root window <" << GetName()
                        << ">" << eom;
    }
    utilmsg(eError) << "cannot remove non-root painter <" << aPainter->GetName() << "> from root window <" << GetName()
                    << ">" << eom;
    return;
}

void KROOTWindow::AddWindow(KWindow* aWindow)
{
    auto* tPad = dynamic_cast<KROOTPad*>(aWindow);
    if (tPad != nullptr) {
        fPads.push_back(tPad);
        tPad->SetWindow(this);
        return;
    }
    utilmsg(eError) << "cannot add <" << aWindow->GetName() << "> to root window <" << GetName() << ">" << eom;
    return;
}
void KROOTWindow::RemoveWindow(KWindow* aWindow)
{
    auto* tPad = dynamic_cast<KROOTPad*>(aWindow);
    if (tPad != nullptr) {
        PadIt tIt;
        for (tIt = fPads.begin(); tIt != fPads.end(); tIt++) {
            if ((*tIt) == tPad) {
                fPads.erase(tIt);
                tPad->ClearWindow(this);
                return;
            }
        }
        utilmsg(eError) << "cannot remove root pad <" << tPad->GetName() << "> from root window <" << GetName() << ">"
                        << eom;
    }
    utilmsg(eError) << "cannot remove <" << aWindow->GetName() << "> from root window <" << GetName() << ">" << eom;
    return;
}

TCanvas* KROOTWindow::GetCanvas()
{
    return fCanvas;
}

TApplication* KROOTWindow::GetApplication()
{
    return fApplication;
}


}  // namespace katrin
