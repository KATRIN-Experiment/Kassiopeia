#include "KSROOTTrackPainter.h"

#include "KSObject.h"
#include "KSReadFileROOT.h"
#include "KSVisualizationMessage.h"
#include "TColor.h"
#include "TGraph.h"
#include "TStyle.h"

#include <limits>

using namespace katrin;
using namespace std;

namespace Kassiopeia
{
KSROOTTrackPainter::KSROOTTrackPainter() :
    fPath(""),
    fBase(""),
    fPlaneNormal(0.0, 1.0, 0.0),
    fPlanePoint(0.0, 0.0, 0.0),
    fSwapAxis(false),
    fPlaneVectorA(0.0, 0.0, 1.0),
    fPlaneVectorB(1.0, 0.0, 0.0),
    fEpsilon(1.0e-10),
    fXAxis("z"),
    fYAxis("y"),
    fStepOutputGroupName("output_step_world"),
    fPositionName("position"),
    fTrackOutputGroupName("output_track_world"),
    fColorVariable("color_variable"),
    fColorMode(eColorFix),
    fColorPalette(eColorDefault),
    fDrawOptions(""),
    fPlotMode(ePlotStep),
    fAxialMirror(false),
    fMultigraph(),
    fBaseColors(),
    fColorVector()
{}
KSROOTTrackPainter::~KSROOTTrackPainter() = default;

std::string KSROOTTrackPainter::GetAxisLabel(KThreeVector anAxis)
{
    if (anAxis.Y() < fEpsilon && anAxis.Y() > -fEpsilon && anAxis.Z() < fEpsilon && anAxis.Z() > -fEpsilon) {
        if (anAxis.X() < 1.0 + fEpsilon && anAxis.X() > 1.0 - fEpsilon) {
            return string("x");
        }
        if (anAxis.X() < -1.0 + fEpsilon && anAxis.X() > -1.0 - fEpsilon) {
            return string("-x");
        }
    }

    if (anAxis.X() < fEpsilon && anAxis.X() > -fEpsilon && anAxis.Z() < fEpsilon && anAxis.Z() > -fEpsilon) {
        if (anAxis.Y() < 1.0 + fEpsilon && anAxis.Y() > 1.0 - fEpsilon) {
            return string("y");
        }
        if (anAxis.Y() < -1.0 + fEpsilon && anAxis.Y() > -1.0 - fEpsilon) {
            return string("-y");
        }
    }

    if (anAxis.X() < fEpsilon && anAxis.X() > -fEpsilon && anAxis.Y() < fEpsilon && anAxis.Y() > -fEpsilon) {
        if (anAxis.Z() < 1.0 + fEpsilon && anAxis.Z() > 1.0 - fEpsilon) {
            return string("z");
        }
        if (anAxis.Z() < -1.0 + fEpsilon && anAxis.Z() > -1.0 - fEpsilon) {
            return string("-z");
        }
    }

    string tLabel;
    std::stringstream ss;
    ss << anAxis.X();
    tLabel += ss.str();
    tLabel += string("/");
    ss.str("");
    ss << anAxis.Y();
    tLabel += ss.str();
    tLabel += string("/");
    ss.str("");
    ss << anAxis.Z();
    tLabel += ss.str();
    return tLabel;
}

void KSROOTTrackPainter::CalculatePlaneCoordinateSystem()
{
    fPlaneNormal = fPlaneNormal.Unit();
    double tDirectionMagX = fabs(fPlaneNormal.X());
    double tDirectionMagY = fabs(fPlaneNormal.Y());
    double tDirectionMagZ = fabs(fPlaneNormal.Z());

    //plane normal looks in x direction
    if (tDirectionMagX >= tDirectionMagY && tDirectionMagX >= tDirectionMagZ) {
        fPlaneVectorA.SetX(0.0);
        fPlaneVectorA.SetY(1.0);
        fPlaneVectorA.SetZ(0.0);

        if (fPlaneNormal.X() > fEpsilon || fPlaneNormal.X() < -1. * fEpsilon) {
            fPlaneVectorA.SetX(-1.0 * fPlaneNormal.Y() / fPlaneNormal.X());
        }

        fPlaneVectorB.SetX(fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y());
        fPlaneVectorB.SetY(fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z());
        fPlaneVectorB.SetZ(fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X());

        fPlaneVectorA = fPlaneVectorA.Unit();
        fPlaneVectorB = fPlaneVectorB.Unit();

        if (fSwapAxis) {
            swap(fPlaneVectorA, fPlaneVectorB);
        }
        vismsg(eInfo) << "Plane vectors are: " << fPlaneVectorA << " and " << fPlaneVectorB << eom;

        if (fPlaneVectorA.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorA.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector A and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        if (fPlaneVectorB.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorB.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector B and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        return;
    }

    //plane normal looks in y direction
    if (tDirectionMagY >= tDirectionMagX && tDirectionMagY >= tDirectionMagZ) {
        fPlaneVectorA.SetX(0.0);
        fPlaneVectorA.SetY(0.0);
        fPlaneVectorA.SetZ(1.0);

        if (fPlaneNormal.Y() > fEpsilon || fPlaneNormal.Y() < -1. * fEpsilon) {
            fPlaneVectorA.SetY(-1.0 * fPlaneNormal.Z() / fPlaneNormal.Y());
        }

        fPlaneVectorB.SetX(fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y());
        fPlaneVectorB.SetY(fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z());
        fPlaneVectorB.SetZ(fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X());

        fPlaneVectorA = fPlaneVectorA.Unit();
        fPlaneVectorB = fPlaneVectorB.Unit();

        if (fSwapAxis) {
            swap(fPlaneVectorA, fPlaneVectorB);
        }
        vismsg(eInfo) << "Plane vectors are: " << fPlaneVectorA << " and " << fPlaneVectorB << eom;

        if (fPlaneVectorA.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorA.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector A and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        if (fPlaneVectorB.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorB.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector B and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        return;
    }

    //plane normal looks in z direction
    if (tDirectionMagZ >= tDirectionMagX && tDirectionMagZ >= tDirectionMagY) {
        fPlaneVectorA.SetX(1.0);
        fPlaneVectorA.SetY(0.0);
        fPlaneVectorA.SetZ(0.0);

        if (fPlaneNormal.Z() > fEpsilon || fPlaneNormal.Z() < -1. * fEpsilon) {
            fPlaneVectorA.SetZ(-1.0 * fPlaneNormal.X() / fPlaneNormal.Z());
        }

        fPlaneVectorB.SetX(fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y());
        fPlaneVectorB.SetY(fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z());
        fPlaneVectorB.SetZ(fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X());

        fPlaneVectorA = fPlaneVectorA.Unit();
        fPlaneVectorB = fPlaneVectorB.Unit();

        if (fSwapAxis) {
            swap(fPlaneVectorA, fPlaneVectorB);
        }
        vismsg(eInfo) << "Plane vectors are: " << fPlaneVectorA << " and " << fPlaneVectorB << eom;

        if (fPlaneVectorA.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorA.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector A and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        if (fPlaneVectorB.Dot(fPlaneNormal) > fEpsilon || fPlaneVectorB.Dot(fPlaneNormal) < -1. * fEpsilon) {
            vismsg(eWarning) << "Scalar product of PlaneVector B and NormalVector is "
                             << fPlaneVectorA.Dot(fPlaneNormal) << eom;
        }
        return;
    }
}

void KSROOTTrackPainter::TransformToPlaneSystem(const KThreeVector aPoint, KTwoVector& aPlanePoint)
{
    //solve aPoint = fPlanePoint + alpha * fPlaneA + beta * fPlaneB for alpha and beta
    double tAlpha, tBeta;

    if ((fPlaneVectorA.X() * fPlaneVectorB.Y() - fPlaneVectorA.Y() * fPlaneVectorB.X()) != 0.0) {
        tAlpha = ((aPoint.X() - fPlanePoint.X()) * fPlaneVectorB.Y() -
                  (aPoint.Y() - fPlanePoint.Y()) * fPlaneVectorB.X()) /
                 (fPlaneVectorA.X() * fPlaneVectorB.Y() - fPlaneVectorA.Y() * fPlaneVectorB.X());

        if (fPlaneVectorB.Y() != 0) {
            //tBeta = (aPoint.Y() - fPlanePoint.Y() - tAlpha * fPlaneVectorA.Y()) / fPlaneVectorB.Y();
            tBeta = (aPoint.X() - fPlanePoint.X() - tAlpha * fPlaneVectorA.X()) * fPlaneVectorB.X() +
                    (aPoint.Y() - fPlanePoint.Y() - tAlpha * fPlaneVectorA.Y()) * fPlaneVectorB.Y() +
                    (aPoint.Z() - fPlanePoint.Z() - tAlpha * fPlaneVectorA.Z()) * fPlaneVectorB.Z();
        }
        else {
            tBeta = (aPoint.X() - fPlanePoint.X() - tAlpha * fPlaneVectorA.X()) / fPlaneVectorB.X();
        }

        aPlanePoint.SetComponents(tAlpha, tBeta);
        vismsg(eInfo) << "Converting " << aPoint << " to " << aPlanePoint << eom;
        return;
    }

    if ((fPlaneVectorA.Y() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.Y()) != 0.0) {
        tAlpha = ((aPoint.Y() - fPlanePoint.Y()) * fPlaneVectorB.Z() - aPoint.Z() * fPlaneVectorB.Y() +
                  fPlanePoint.Z() * fPlaneVectorB.Y()) /
                 (fPlaneVectorA.Y() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.Y());

        if (fPlaneVectorB.Z() != 0) {
            tBeta = (aPoint.Z() - fPlanePoint.Z() - tAlpha * fPlaneVectorA.Z()) / fPlaneVectorB.Z();
        }
        else {
            tBeta = (aPoint.Y() - fPlanePoint.Y() - tAlpha * fPlaneVectorA.Y()) / fPlaneVectorB.Y();
        }

        aPlanePoint.SetComponents(tAlpha, tBeta);
        vismsg(eInfo) << "Converting " << aPoint << " to " << aPlanePoint << eom;
        return;
    }

    if ((fPlaneVectorA.X() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.X()) != 0.0) {
        tAlpha = ((aPoint.X() - fPlanePoint.X()) * fPlaneVectorB.Z() - aPoint.Z() * fPlaneVectorB.X() +
                  fPlanePoint.Z() * fPlaneVectorB.X()) /
                 (fPlaneVectorA.X() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.X());

        if (fPlaneVectorB.Z() != 0) {
            tBeta = (aPoint.Z() - fPlanePoint.Z() - tAlpha * fPlaneVectorA.Z()) / fPlaneVectorB.Z();
        }
        else {
            tBeta = (aPoint.X() - fPlanePoint.X() - tAlpha * fPlaneVectorA.X()) / fPlaneVectorB.X();
        }

        aPlanePoint.SetComponents(tAlpha, tBeta);
        vismsg(eInfo) << "Converting " << aPoint << " to " << aPlanePoint << eom;
        return;
    }

    vismsg(eWarning) << "this should never be called - problem in TransformToPlaneSystem function" << eom;
    return;
}


void KSROOTTrackPainter::Render()
{
    CalculatePlaneCoordinateSystem();

    fMultigraph = new TMultiGraph();

    KRootFile* tRootFile = KRootFile::CreateOutputRootFile(fBase);
    if (!fPath.empty()) {
        tRootFile->AddToPaths(fPath);
    }

    KSReadFileROOT tReader;
    if (!tReader.TryFile(tRootFile)) {
        vismsg(eWarning) << "Could not open file <" << tRootFile->GetName() << ">" << eom;
        return;
    }

    tReader.OpenFile(tRootFile);

    CreateColors(tReader);
    auto tColorIterator = fColorVector.begin();

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    KSReadStepROOT& tStepReader = tReader.GetStep();

    if (fPlotMode == ePlotStep) {
        KSReadObjectROOT& tStepGroup = tStepReader.GetObject(fStepOutputGroupName);
        auto& tPosition = tStepGroup.Get<KSThreeVector>(fPositionName);

        for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
            vismsg(eDebug) << "Analyzing run <" << tRunReader.GetRunIndex() << "> with events from <"
                           << tRunReader.GetFirstEventIndex() << "> to <" << tRunReader.GetLastEventIndex() << ">"
                           << eom;
            for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                 tEventReader++) {
                vismsg(eDebug) << "Analyzing event <" << tEventReader.GetEventIndex() << "> with tracks from <"
                               << tEventReader.GetFirstTrackIndex() << "> to <" << tEventReader.GetLastTrackIndex()
                               << ">" << eom;
                for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                     tTrackReader++) {
                    vismsg(eDebug) << "Analyzing track <" << tTrackReader.GetTrackIndex() << "> with steps from <"
                                   << tTrackReader.GetFirstStepIndex() << "> to <" << tTrackReader.GetLastStepIndex()
                                   << ">" << eom;
                    TGraph* myGraph = nullptr;
                    if (fColorMode == eColorTrack || fColorMode == eColorFix) {
                        myGraph = new TGraph();
                        if (tColorIterator == fColorVector.end()) {
                            vismsg(eError) << "color vector has to less entries, something is wrong!" << eom;
                        }
                        myGraph->SetLineColor(*tColorIterator);
                        tColorIterator++;
                    }

                    for (tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex();
                         tStepReader++) {
                        if (tStepGroup.Valid()) {
                            double tXValue = 0.;
                            if (fXAxis == string("x") || fXAxis == string("X")) {
                                tXValue = tPosition.Value().X();
                            }
                            if (fXAxis == string("y") || fXAxis == string("Y")) {
                                tXValue = tPosition.Value().Y();
                            }
                            if (fXAxis == string("z") || fXAxis == string("Z")) {
                                tXValue = tPosition.Value().Z();
                            }
                            if (fXAxis == string("a") || fXAxis == string("A")) {
                                KTwoVector tPlanePoint;
                                TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                                tXValue = tPlanePoint.X();
                            }
                            if (fXAxis == string("b") || fXAxis == string("B")) {
                                KTwoVector tPlanePoint;
                                TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                                tXValue = tPlanePoint.Y();
                            }
                            double tYValue = 0.;
                            if (fYAxis == string("x") || fYAxis == string("X")) {
                                tYValue = tPosition.Value().X();
                            }
                            if (fYAxis == string("y") || fYAxis == string("Y")) {
                                tYValue = tPosition.Value().Y();
                            }
                            if (fYAxis == string("z") || fYAxis == string("Z")) {
                                tYValue = tPosition.Value().Z();
                            }
                            if (fYAxis == string("r") || fYAxis == string("R")) {
                                tYValue = tPosition.Value().Perp();
                            }
                            if (fYAxis == string("a") || fYAxis == string("A")) {
                                KTwoVector tPlanePoint;
                                TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                                tYValue = tPlanePoint.X();
                            }
                            if (fYAxis == string("b") || fYAxis == string("B")) {
                                KTwoVector tPlanePoint;
                                TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                                tYValue = tPlanePoint.Y();
                            }

                            if (fColorMode == eColorStep) {
                                //create one graph for each point (one graph can only have one color)
                                myGraph = new TGraph();
                                myGraph->SetPoint(myGraph->GetN(), tXValue, tYValue);
                                if (tColorIterator == fColorVector.end()) {
                                    vismsg(eError) << "color vector has to less entries, something is wrong!" << eom;
                                }
                                myGraph->SetMarkerColor(*tColorIterator);
                                tColorIterator++;
                                if (myGraph->GetN() > 0) {
                                    fMultigraph->Add(myGraph);
                                }
                            }
                            if (fColorMode == eColorTrack || fColorMode == eColorFix) {
                                myGraph->SetPoint(myGraph->GetN(), tXValue, tYValue);
                            }
                        }
                    }

                    if (fColorMode == eColorTrack || fColorMode == eColorFix) {
                        if (myGraph->GetN() > 0) {
                            fMultigraph->Add(myGraph);
                        }

                        //if axial mirror is set, another graph is created with the same points, put y has a changed sign
                        if (fAxialMirror) {
                            auto* myMirroredGraph = new TGraph();
                            myMirroredGraph->SetLineColor(myGraph->GetLineColor());
                            double tX, tY;
                            for (int tIndex = 0; tIndex < myGraph->GetN(); tIndex++) {
                                myGraph->GetPoint(tIndex, tX, tY);
                                myMirroredGraph->SetPoint(tIndex, tX, -1.0 * tY);
                            }
                            if (myMirroredGraph->GetN() > 0) {
                                fMultigraph->Add(myMirroredGraph);
                            }
                        }
                    }
                }
            }
        }
    }

    if (fPlotMode == ePlotTrack) {
        KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject(fTrackOutputGroupName);
        auto& tPosition = tTrackGroup.Get<KSThreeVector>(fPositionName);
        for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
            for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                 tEventReader++) {

                for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                     tTrackReader++) {
                    TGraph* myGraph;
                    myGraph = new TGraph();

                    if (fColorMode == eColorTrack || fColorMode == eColorFix) {
                        if (tColorIterator == fColorVector.end()) {
                            vismsg(eError) << "color vector has to less entries, something is wrong!" << eom;
                        }
                        myGraph->SetMarkerColor(*tColorIterator);
                        tColorIterator++;
                    }
                    double tXValue = 0.;
                    if (fXAxis == string("x") || fXAxis == string("X")) {
                        tXValue = tPosition.Value().X();
                    }
                    if (fXAxis == string("y") || fXAxis == string("Y")) {
                        tXValue = tPosition.Value().Y();
                    }
                    if (fXAxis == string("z") || fXAxis == string("Z")) {
                        tXValue = tPosition.Value().Z();
                    }
                    if (fXAxis == string("a") || fXAxis == string("A")) {
                        KTwoVector tPlanePoint;
                        TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                        tXValue = tPlanePoint.X();
                    }
                    if (fXAxis == string("b") || fXAxis == string("B")) {
                        KTwoVector tPlanePoint;
                        TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                        tXValue = tPlanePoint.Y();
                    }
                    double tYValue = 0.;
                    if (fYAxis == string("x") || fYAxis == string("X")) {
                        tYValue = tPosition.Value().X();
                    }
                    if (fYAxis == string("y") || fYAxis == string("Y")) {
                        tYValue = tPosition.Value().Y();
                    }
                    if (fYAxis == string("z") || fYAxis == string("Z")) {
                        tYValue = tPosition.Value().Z();
                    }
                    if (fYAxis == string("r") || fYAxis == string("R")) {
                        tYValue = tPosition.Value().Perp();
                    }
                    if (fYAxis == string("a") || fYAxis == string("A")) {
                        KTwoVector tPlanePoint;
                        TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                        tYValue = tPlanePoint.X();
                    }
                    if (fYAxis == string("b") || fYAxis == string("B")) {
                        KTwoVector tPlanePoint;
                        TransformToPlaneSystem(tPosition.Value(), tPlanePoint);
                        tYValue = tPlanePoint.Y();
                    }

                    myGraph->SetPoint(myGraph->GetN(), tXValue, tYValue);
                    if (myGraph->GetN() > 0) {
                        fMultigraph->Add(myGraph);
                    }
                }
            }
        }
    }

    if (fMultigraph->GetListOfGraphs() != nullptr)
        vismsg(eNormal) << "root track painter has " << fMultigraph->GetListOfGraphs()->GetSize() << " elements" << eom;
    else
        vismsg(eWarning) << "root track painter found no data to plot!" << eom;

    tReader.CloseFile();
    delete tRootFile;

    return;
}

void KSROOTTrackPainter::Display()
{
    if ((fDisplayEnabled == true) && (fMultigraph->GetListOfGraphs() != nullptr)) {
        if (!fDrawOptions.empty()) {
            fMultigraph->Draw(fDrawOptions.c_str());
        }
        else if (fPlotMode == ePlotStep) {
            if (fColorMode == eColorStep) {
                fMultigraph->Draw("P");
            }
            else {
                fMultigraph->Draw("L");
            }
        }
        else if (fPlotMode == ePlotTrack) {
            fMultigraph->Draw("P");
        }
    }

    return;
}

void KSROOTTrackPainter::Write()
{
    if (fWriteEnabled == true) {
        return;
    }
    return;
}

void KSROOTTrackPainter::CreateColors(KSReadFileROOT& aReader)
{
    KSReadRunROOT& tRunReader = aReader.GetRun();
    KSReadEventROOT& tEventReader = aReader.GetEvent();
    KSReadTrackROOT& tTrackReader = aReader.GetTrack();
    KSReadStepROOT& tStepReader = aReader.GetStep();

    //getting number of tracks/steps in file
    size_t tNumberOfTracks = 0;
    size_t tNumberOfSteps = 0;
    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                tNumberOfTracks++;

                for (tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex();
                     tStepReader++) {
                    tNumberOfSteps++;
                }
            }
        }
    }

    if (fColorPalette == eColorFPDRings) {
        while ((fColorVector.size() < tNumberOfTracks && fColorMode == eColorTrack) ||
               (fColorVector.size() < tNumberOfSteps && fColorMode == eColorStep)) {
            //rainbow scheme
            fColorVector.push_back(kBlack);
            fColorVector.push_back(kViolet + 7);
            fColorVector.push_back(kBlue + 2);
            fColorVector.push_back(kAzure + 2);
            fColorVector.push_back(kAzure + 10);
            fColorVector.push_back(kTeal + 7);
            fColorVector.push_back(kGreen + 1);
            fColorVector.push_back(kSpring - 3);
            fColorVector.push_back(kSpring + 10);
            fColorVector.push_back(kYellow);
            fColorVector.push_back(kOrange - 3);
            fColorVector.push_back(kOrange + 7);
            fColorVector.push_back(kRed);
            fColorVector.push_back(kRed + 2);
        }
    }

    if (fColorMode == eColorFix) {
        Color_t tFixColor(kRed);
        if (fBaseColors.size() > 0) {
            TColor tTempColor;
            tFixColor = tTempColor.GetColor(fBaseColors.at(0).first.GetRed(),
                                            fBaseColors.at(0).first.GetGreen(),
                                            fBaseColors.at(0).first.GetBlue());
        }
        while (fColorVector.size() < tNumberOfTracks) {
            fColorVector.push_back(tFixColor);
        }
    }

    if (fColorPalette == eColorCustom || fColorPalette == eColorDefault) {

        int tColorBins = 100;
        size_t tNumberBaseColors = fBaseColors.size();

        double tRed[tNumberBaseColors];
        double tGreen[tNumberBaseColors];
        double tBlue[tNumberBaseColors];
        double tFraction[tNumberBaseColors];

        for (size_t tIndex = 0; tIndex < tNumberBaseColors; tIndex++) {
            tRed[tIndex] = fBaseColors.at(tIndex).first.GetRed();
            tGreen[tIndex] = fBaseColors.at(tIndex).first.GetGreen();
            tBlue[tIndex] = fBaseColors.at(tIndex).first.GetBlue();
            tFraction[tIndex] = fBaseColors.at(tIndex).second;
            if (tFraction[tIndex] == -1.0) {
                tFraction[tIndex] = tIndex / (double) (tNumberBaseColors - 1);
            }
        }

        int tMinColor = TColor::CreateGradientColorTable(tNumberBaseColors, tFraction, tRed, tGreen, tBlue, tColorBins);
        int tMaxColor = tMinColor + tColorBins - 1;

        if (fColorPalette == eColorDefault) {
            tMinColor = 51;   //purple
            tMaxColor = 100;  //red
        }

        //        int tPalette[tColorBins];
        //        for ( int i = 0; i<tColorBins; i++)
        //        {
        //          tPalette[i] = tMinColor + i;
        //        }
        //        gStyle->SetPalette( tColorBins, tPalette );


        double tColorVariableMax(-1.0 * std::numeric_limits<double>::max());
        double tColorVariableMin(std::numeric_limits<double>::max());

        if (fColorMode == eColorTrack) {
            if (tNumberOfTracks == 1) {
                fColorVector.push_back(tMaxColor);
                return;
            }
            //get track group and color variable
            KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject(fTrackOutputGroupName);

            //find min and max of color variable
            for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
                for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                     tEventReader++) {
                    for (tTrackReader = tEventReader.GetFirstTrackIndex();
                         tTrackReader <= tEventReader.GetLastTrackIndex();
                         tTrackReader++) {
                        double tColorVariable = 0.;
                        if (tTrackGroup.Exists<KSString>(fColorVariable)) {
                            string tColorKey = tTrackGroup.Get<KSString>(fColorVariable).Value();
                            if (fColorIndex.find(tColorKey) == fColorIndex.end())
                                fColorIndex[tColorKey] = fColorIndex.size();
                            tColorVariable = fColorIndex[tColorKey];
                        }
                        else if (tTrackGroup.Exists<KSDouble>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSDouble>(fColorVariable).Value();
                        }
                        else if (tTrackGroup.Exists<KSInt>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSInt>(fColorVariable).Value();
                        }
                        else if (tTrackGroup.Exists<KSUInt>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSUInt>(fColorVariable).Value();
                        }
                        else {
                            vismsg(eError) << "Color variable is of unsupported type" << eom;
                        }


                        if (tColorVariable > tColorVariableMax) {
                            tColorVariableMax = tColorVariable;
                        }
                        if (tColorVariable < tColorVariableMin) {
                            tColorVariableMin = tColorVariable;
                        }
                    }
                }
            }
            vismsg(eInfo) << "Range of track color variable is from < " << tColorVariableMin << " > to < "
                          << tColorVariableMax << " >" << eom;

            for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
                for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                     tEventReader++) {
                    for (tTrackReader = tEventReader.GetFirstTrackIndex();
                         tTrackReader <= tEventReader.GetLastTrackIndex();
                         tTrackReader++) {
                        double tColorVariable = 0.;
                        if (tTrackGroup.Exists<KSString>(fColorVariable)) {
                            string tColorKey = tTrackGroup.Get<KSString>(fColorVariable).Value();
                            if (fColorIndex.find(tColorKey) == fColorIndex.end())
                                fColorIndex[tColorKey] = fColorIndex.size();
                            tColorVariable = fColorIndex[tColorKey];
                        }
                        else if (tTrackGroup.Exists<KSDouble>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSDouble>(fColorVariable).Value();
                        }
                        else if (tTrackGroup.Exists<KSInt>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSInt>(fColorVariable).Value();
                        }
                        else if (tTrackGroup.Exists<KSUInt>(fColorVariable)) {
                            tColorVariable = tTrackGroup.Get<KSUInt>(fColorVariable).Value();
                        }
                        else {
                            vismsg(eError) << "Color variable is of unsupported type" << eom;
                        }
                        double tCurrentColor =
                            tMinColor + ((tMaxColor - tMinColor) * (tColorVariable - tColorVariableMin) /
                                         (tColorVariableMax - tColorVariableMin));
                        fColorVector.push_back(tCurrentColor);
                    }
                }
            }
        }

        if (fColorMode == eColorStep) {
            //get step group and color variable
            KSReadObjectROOT& tStepGroup = tStepReader.GetObject(fStepOutputGroupName);

            //find min and max of color variable
            for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
                for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                     tEventReader++) {
                    for (tTrackReader = tEventReader.GetFirstTrackIndex();
                         tTrackReader <= tEventReader.GetLastTrackIndex();
                         tTrackReader++) {
                        for (tStepReader = tTrackReader.GetFirstStepIndex();
                             tStepReader <= tTrackReader.GetLastStepIndex();
                             tStepReader++) {
                            if (tStepGroup.Valid()) {
                                double tColorVariable = 0.;
                                if (tStepGroup.Exists<KSString>(fColorVariable)) {
                                    string tColorKey = tStepGroup.Get<KSString>(fColorVariable).Value();
                                    if (fColorIndex.find(tColorKey) == fColorIndex.end())
                                        fColorIndex[tColorKey] = fColorIndex.size();
                                    tColorVariable = fColorIndex[tColorKey];
                                }
                                else if (tStepGroup.Exists<KSDouble>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSDouble>(fColorVariable).Value();
                                }
                                else if (tStepGroup.Exists<KSInt>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSInt>(fColorVariable).Value();
                                }
                                else if (tStepGroup.Exists<KSUInt>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSUInt>(fColorVariable).Value();
                                }
                                else {
                                    vismsg(eError) << "Color variable is of unsupported type" << eom;
                                }

                                if (tColorVariable > tColorVariableMax) {
                                    tColorVariableMax = tColorVariable;
                                }
                                if (tColorVariable < tColorVariableMin) {
                                    tColorVariableMin = tColorVariable;
                                }
                            }
                        }
                    }
                }
            }

            vismsg(eInfo) << "Range of step color variable is from < " << tColorVariableMin << " > to < "
                          << tColorVariableMax << " >" << eom;

            for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
                for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                     tEventReader++) {
                    for (tTrackReader = tEventReader.GetFirstTrackIndex();
                         tTrackReader <= tEventReader.GetLastTrackIndex();
                         tTrackReader++) {
                        for (tStepReader = tTrackReader.GetFirstStepIndex();
                             tStepReader <= tTrackReader.GetLastStepIndex();
                             tStepReader++) {
                            if (tStepGroup.Valid()) {
                                double tColorVariable = 0.;
                                if (tStepGroup.Exists<KSString>(fColorVariable)) {
                                    string tColorKey = tStepGroup.Get<KSString>(fColorVariable).Value();
                                    if (fColorIndex.find(tColorKey) == fColorIndex.end())
                                        fColorIndex[tColorKey] = fColorIndex.size();
                                    tColorVariable = fColorIndex[tColorKey];
                                }
                                else if (tStepGroup.Exists<KSDouble>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSDouble>(fColorVariable).Value();
                                }
                                else if (tStepGroup.Exists<KSInt>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSInt>(fColorVariable).Value();
                                }
                                else if (tStepGroup.Exists<KSUInt>(fColorVariable)) {
                                    tColorVariable = tStepGroup.Get<KSUInt>(fColorVariable).Value();
                                }
                                else {
                                    vismsg(eError) << "Color variable is of unsupported type" << eom;
                                }
                                double tCurrentColor =
                                    tMinColor + ((tMaxColor - tMinColor) * (tColorVariable - tColorVariableMin) /
                                                 (tColorVariableMax - tColorVariableMin));
                                fColorVector.push_back(tCurrentColor);
                            }
                        }
                    }
                }
            }
        }
    }
}

double KSROOTTrackPainter::GetXMin()
{
    double tMin(std::numeric_limits<double>::max());
    TList* tGraphList = fMultigraph->GetListOfGraphs();
    if (tGraphList != nullptr) {
        for (int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++) {
            auto* tGraph = dynamic_cast<TGraph*>(tGraphList->At(tIndex));
            double* tX = tGraph->GetX();
            for (int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++) {
                if (tX[tIndexArray] < tMin) {
                    tMin = tX[tIndexArray];
                }
            }
        }
    }
    return tMin;
}
double KSROOTTrackPainter::GetXMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    TList* tGraphList = fMultigraph->GetListOfGraphs();
    if (tGraphList != nullptr) {
        for (int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++) {
            auto* tGraph = dynamic_cast<TGraph*>(tGraphList->At(tIndex));
            double* tX = tGraph->GetX();
            for (int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++) {
                if (tX[tIndexArray] > tMax) {
                    tMax = tX[tIndexArray];
                }
            }
        }
    }
    return tMax;
}

double KSROOTTrackPainter::GetYMin()
{
    double tMin(std::numeric_limits<double>::max());
    TList* tGraphList = fMultigraph->GetListOfGraphs();
    if (tGraphList != nullptr) {
        for (int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++) {
            auto* tGraph = dynamic_cast<TGraph*>(tGraphList->At(tIndex));
            double* tY = tGraph->GetY();
            for (int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++) {
                if (tY[tIndexArray] < tMin) {
                    tMin = tY[tIndexArray];
                }
            }
        }
    }
    return tMin;
}
double KSROOTTrackPainter::GetYMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    TList* tGraphList = fMultigraph->GetListOfGraphs();
    if (tGraphList != nullptr) {
        for (int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++) {
            auto* tGraph = dynamic_cast<TGraph*>(tGraphList->At(tIndex));
            double* tY = tGraph->GetY();
            for (int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++) {
                if (tY[tIndexArray] > tMax) {
                    tMax = tY[tIndexArray];
                }
            }
        }
    }
    return tMax;
}

std::string KSROOTTrackPainter::GetXAxisLabel()
{
    return fXAxis;
}

std::string KSROOTTrackPainter::GetYAxisLabel()
{
    return fYAxis;
}

}  // namespace Kassiopeia
