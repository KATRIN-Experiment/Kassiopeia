#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KConst.h"
#include "KElementProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KPrintProcessor.hh"
#include "KRandom.h"
#include "KSEvent.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSObject.h"
#include "KSRootMagneticField.h"
#include "KTagProcessor.hh"
#include "KToolbox.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"
#include "TApplication.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TH3D.h"
#include "TMultiGraph.h"

#include <sstream>

using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    // read in xml file
    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine(argc, argv);

    KXMLTokenizer tXMLTokenizer;
    KVariableProcessor tVariableProcessor(tCommandLine.GetVariables());
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter(&tXMLTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tIncludeProcessor.InsertAfter(&tFormulaProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tConditionProcessor.InsertAfter(&tLoopProcessor);
    tPrintProcessor.InsertAfter(&tConditionProcessor);
    tTagProcessor.InsertAfter(&tPrintProcessor);
    tElementProcessor.InsertAfter(&tTagProcessor);

    KTextFile tInputFile;
    tInputFile.AddToBases("TestRampedField.xml");
    tInputFile.AddToPaths(string(CONFIG_DEFAULT_DIR) + string("/Validation"));
    tXMLTokenizer.ProcessFile(&tInputFile);

    // read in command line parameters
    if (argc <= 2) {
        mainmsg(eWarning) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestRampedField <b_field> <e_field>" << ret;
        mainmsg << ret;
        mainmsg << "available field components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestRampedField.xml"
                << ret;
        mainmsg << eom;
        return -1;
    }

    string tPathB(argv[1]);
    string tPathE(argv[2]);

    unsigned int tCount = 1000;
    double tStep = 0.001;

    // initialize magnetic field
    KSMagneticField* tRootMagneticField = nullptr;
    try {
        tRootMagneticField = getMagneticField(tPathB);
    }
    catch (...) {
    }
    if (tRootMagneticField) {
        tRootMagneticField->Initialize();
        tRootMagneticField->Activate();
    }

    // initialize electric field
    KSElectricField* tRootElectricField = nullptr;
    try {
        tRootElectricField = getElectricField(tPathE);
    }
    catch (...) {
    }

    if (tRootElectricField) {
        tRootElectricField->Initialize();
        tRootElectricField->Activate();
    }

    KThreeVector tPoint, tMagneticPotential, tElectricField, tMagneticField;
    double tElectricPotential;
    KThreeMatrix tMagneticGradient;

    // initialize root
    TApplication tApplication("TestRampedField", nullptr, nullptr);

    auto* tMagneticFieldXGraph = new TGraph();
    auto* tMagneticFieldYGraph = new TGraph();
    auto* tMagneticFieldZGraph = new TGraph();

    auto* tElectricFieldXGraph = new TGraph();
    auto* tElectricFieldYGraph = new TGraph();
    auto* tElectricFieldZGraph = new TGraph();

    for (unsigned int i = 0; i < tCount; i++) {
        double tTime = i * tStep;

        tPoint.SetComponents(1, 0, 0);

        if (tRootMagneticField) {
            tRootMagneticField->CalculateField(tPoint, tTime, tMagneticField);
            tRootMagneticField->CalculateGradient(tPoint, tTime, tMagneticGradient);

            tMagneticFieldXGraph->SetPoint(tMagneticFieldXGraph->GetN(), tTime, tMagneticField.X());
            tMagneticFieldYGraph->SetPoint(tMagneticFieldYGraph->GetN(), tTime, tMagneticField.Y());
            tMagneticFieldZGraph->SetPoint(tMagneticFieldZGraph->GetN(), tTime, tMagneticField.Z());
        }

        if (tRootElectricField) {
            tRootElectricField->CalculatePotential(tPoint, tTime, tElectricPotential);
            tRootElectricField->CalculateField(tPoint, tTime, tElectricField);

            tElectricFieldXGraph->SetPoint(tElectricFieldXGraph->GetN(), tTime, tElectricField.X());
            tElectricFieldYGraph->SetPoint(tElectricFieldYGraph->GetN(), tTime, tElectricField.Y());
            tElectricFieldZGraph->SetPoint(tElectricFieldZGraph->GetN(), tTime, tElectricField.Z());
        }

        std::cout << "Point: " << tPoint << "\tTime:" << tTime << std::endl;
        if (tRootMagneticField) {
            std::cout << "  Magnetic Field:     " << tMagneticField << std::endl;
            std::cout << "  Magnetic Gradient:  " << tMagneticGradient << std::endl;
        }
        if (tRootElectricField) {
            std::cout << "  Electric Potential: " << tElectricPotential << std::endl;
            std::cout << "  Electric Field:     " << tElectricField << std::endl;
        }
    }

    // show plots
    auto* tCanvas = new TCanvas("canvas", "TestRampedField");
    tCanvas->Divide(2, 3);

    tCanvas->cd(1);
    tMagneticFieldXGraph->SetLineColor(kBlue - 2);
    tMagneticFieldXGraph->SetTitle("Magnetic Field X");
    tMagneticFieldXGraph->GetXaxis()->SetTitle("Time t [s]");
    tMagneticFieldXGraph->GetYaxis()->SetTitle("Magnetic Field X [T]");
    tMagneticFieldXGraph->Write();
    tMagneticFieldXGraph->Draw("AL");

    tCanvas->cd(3);
    tMagneticFieldYGraph->SetLineColor(kGreen - 2);
    tMagneticFieldYGraph->SetTitle("Magnetic Field Y");
    tMagneticFieldYGraph->GetXaxis()->SetTitle("Time t [s]");
    tMagneticFieldYGraph->GetYaxis()->SetTitle("Magnetic Field Y [T]");
    tMagneticFieldYGraph->Write();
    tMagneticFieldYGraph->Draw("AL");

    tCanvas->cd(5);
    tMagneticFieldZGraph->SetLineColor(kRed - 2);
    tMagneticFieldZGraph->SetTitle("Magnetic Field Z");
    tMagneticFieldZGraph->GetXaxis()->SetTitle("Time t [s]");
    tMagneticFieldZGraph->GetYaxis()->SetTitle("Magnetic Field Z [T]");
    tMagneticFieldZGraph->Write();
    tMagneticFieldZGraph->Draw("AL");

    tCanvas->cd(2);
    tElectricFieldXGraph->SetLineColor(kBlue - 2);
    tElectricFieldXGraph->SetTitle("Electric Field X");
    tElectricFieldXGraph->GetXaxis()->SetTitle("Time t [s]");
    tElectricFieldXGraph->GetYaxis()->SetTitle("Electric Field X [V/m]");
    tElectricFieldXGraph->Write();
    tElectricFieldXGraph->Draw("AL");

    tCanvas->cd(4);
    tElectricFieldYGraph->SetLineColor(kGreen - 2);
    tElectricFieldYGraph->SetTitle("Electric Field Y");
    tElectricFieldYGraph->GetXaxis()->SetTitle("Time t [s]");
    tElectricFieldYGraph->GetYaxis()->SetTitle("Electric Field Y [V/m]");
    tElectricFieldYGraph->Write();
    tElectricFieldYGraph->Draw("AL");

    tCanvas->cd(6);
    tElectricFieldZGraph->SetLineColor(kRed - 2);
    tElectricFieldZGraph->SetTitle("Electric Field Z");
    tElectricFieldZGraph->GetXaxis()->SetTitle("Time t [s]");
    tElectricFieldZGraph->GetYaxis()->SetTitle("Electric Field Z [V/m]");
    tElectricFieldZGraph->Write();
    tElectricFieldZGraph->Draw("AL");

    tCanvas->SaveAs("TestRampedField.pdf");

    tApplication.Run();

    delete tMagneticFieldXGraph;
    delete tMagneticFieldYGraph;
    delete tMagneticFieldZGraph;
    delete tElectricFieldXGraph;
    delete tElectricFieldYGraph;
    delete tElectricFieldZGraph;
    delete tCanvas;

    // deinitialize magnetic field
    if (tRootMagneticField) {
        tRootMagneticField->Deactivate();
        tRootMagneticField->Deinitialize();
    }

    // deinitialize electric field
    if (tRootElectricField) {
        tRootElectricField->Deactivate();
        tRootElectricField->Deinitialize();
    }


    return 0;
}
