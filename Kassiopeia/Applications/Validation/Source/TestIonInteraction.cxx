#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KConst.h"
#include "KElementProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KSIntCalculatorIon.h"
#include "KSIntDensityConstant.h"
#include "KSIntScattering.h"
#include "KSMainMessage.h"
#include "KSParticleFactory.h"
#include "KTagProcessor.hh"
#include "KToolbox.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"
#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TLegend.h"

#include <sstream>
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main(int /*anArgc*/, char** /*anArgv*/)
{
    // read in xml file
    KXMLTokenizer tXMLTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter(&tXMLTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tIncludeProcessor.InsertAfter(&tFormulaProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tConditionProcessor.InsertAfter(&tLoopProcessor);
    tTagProcessor.InsertAfter(&tConditionProcessor);
    tElementProcessor.InsertAfter(&tTagProcessor);

    auto* tDensityCalc = new KSIntDensityConstant();
    tDensityCalc->SetPressure(1.e-4);    // pascal (*100 for mbar)
    tDensityCalc->SetTemperature(300.);  // kelvin

    KSIntCalculator* IonIntCalculator = new KSIntCalculatorIon();
    IonIntCalculator->SetName("ion_ionisation");
    IonIntCalculator->SetTag("ion");


    // get stuff from toolbox
    auto* tScattering = new KSIntScattering();
    tScattering->SetDensity(tDensityCalc);
    tScattering->SetSplit(false);

    tScattering->AddCalculator(IonIntCalculator);

    tScattering->Initialize();

    // initialize root
    TApplication tApplication("Test Interaction", nullptr, nullptr);

    TCanvas tCrossSectionCanvas("crosssection_canvas", "Cross section on H_2");
    TLegend tCrossSectionLegend(0.70, 0.15, 0.85, 0.5);
    TGraph tCrossSectionGraphList[4];
    //tCrossSectionGraphList[0] = TGraph tHCrossSectionGraph;
    //tCrossSectionGraphList[1] = TGraph tH2CrossSectionGraph;
    //tCrossSectionGraphList[2] = TGraph tH3CrossSectionGraph;
    //tCrossSectionGraphList[3] = TGraph tHMinusCrossSectionGraph;

    //Compare simulation with data from
    //T.Tabata and T.Shirai, Analytic Cross Sections for Collisions of H+,H2+,H3+,H,H2 ,and H- with Hydrogen Molecules,
    //Atomic Data and Nuclear Data Tables 76,1 (2000).
    //http://www-jt60.naka.qst.go.jp/english/JEAMDL/code/00000.html

    //H+ on H2 data
    double XDATA1[24] = {75.00000,   100.00000,  133.40000,  177.80000,  237.00000,  316.00000,
                         421.00000,  562.00000,  750.00000,  1000.00000, 1334.00000, 1778.00000,
                         2371.00000, 3162.00000, 4217.00000, 5623.00000, 7499.00000, 10000.0000,
                         15000.0000, 20000.0000, 30000.0000, 50000.0000, 70000.0000, 100000.000};
    double YDATA1[24] = {1.020E-19, 1.700E-19, 2.500E-19, 3.700E-19, 5.400E-19, 7.700E-19, 1.090E-18, 1.620E-18,
                         2.300E-18, 3.300E-18, 4.800E-18, 6.900E-18, 9.800E-18, 1.400E-17, 2.000E-17, 2.800E-17,
                         4.000E-17, 5.500E-17, 8.800E-17, 1.180E-16, 1.670E-16, 2.120E-16, 2.160E-16, 1.960E-16};
    TGraph data_HPlus(24, XDATA1, YDATA1);

    //H2+ on H2 data
    double XDATA2[26] = {31.60000,   42.20000,   56.20000,   75.00000,   100.00000,  133.40000,  177.80000,
                         237.00000,  316.00000,  421.00000,  562.00000,  750.00000,  1000.00000, 1334.00000,
                         1778.00000, 2371.00000, 3162.00000, 4217.00000, 5623.00000, 7499.00000, 10000.0000,
                         2.6E+04,    5.7E+04,    7.5E+04,    8.6E+04,    1.0E+05};
    double YDATA2[26] = {1.000E-20, 8.000E-20, 2.100E-19, 4.800E-19, 8.400E-19, 1.200E-18, 1.700E-18,
                         2.400E-18, 3.350E-18, 4.700E-18, 6.400E-18, 9.100E-18, 1.240E-17, 1.700E-17,
                         2.300E-17, 3.100E-17, 4.100E-17, 5.500E-17, 7.300E-17, 9.500E-17, 1.270E-16,
                         2.240E-16, 3.340E-16, 3.580E-16, 3.530E-16, 3.800E-16};
    TGraph data_H2Plus(26, XDATA2, YDATA2);

    //H3+ on H2 data
    double XDATA3[21] = {75.00000,   100.00000,  133.40000,  177.80000,  237.00000,  316.00000,  422.00000,
                         562.00000,  750.00000,  1000.00000, 1334.00000, 1778.00000, 2371.00000, 3162.00000,
                         4217.00000, 5623.00000, 7499.00000, 10000.0000, 3.0E+04,    6.0E+04,    1.0E+05};
    double YDATA3[21] = {1.230E-18, 2.080E-18, 3.050E-18, 4.100E-18, 5.250E-18, 6.700E-18, 8.250E-18,
                         1.000E-17, 1.200E-17, 1.430E-17, 1.740E-17, 2.180E-17, 2.730E-17, 3.500E-17,
                         4.700E-17, 6.100E-17, 8.100E-17, 1.060E-16, 2.400E-16, 3.600E-16, 4.000E-16};
    TGraph data_H3Plus(21, XDATA3, YDATA3);

    double x_HMinus[35] = {2.37000,    3.16000,    4.22000,    5.62000,    7.50000,    10.00000,   13.34000,
                           17.78000,   23.70000,   31.60000,   42.20000,   56.20000,   75.00000,   100.00000,
                           133.40000,  177.80000,  237.00000,  316.00000,  421.00000,  562.00000,  750.00000,
                           1000.00000, 1334.00000, 1778.00000, 2371.00000, 3162.00000, 4217.00000, 5623.00000,
                           7499.00000, 10000.0000, 1.5E+04,    2.0E+04,    3.0E+04,    4.0E+04,    5.0E+04};
    double y_HMinus[35] = {9.500E-18, 7.800E-17, 1.590E-16, 2.370E-16, 2.960E-16, 3.420E-16, 3.730E-16,
                           3.940E-16, 4.100E-16, 4.250E-16, 4.350E-16, 4.430E-16, 4.550E-16, 4.850E-16,
                           5.300E-16, 5.900E-16, 6.600E-16, 7.350E-16, 8.200E-16, 9.000E-16, 9.700E-16,
                           1.020E-15, 1.080E-15, 1.130E-15, 1.180E-15, 1.210E-15, 1.220E-15, 1.230E-15,
                           1.250E-15, 1.270E-15, 8.70E-16,  7.77E-16,  7.38E-16,  5.80E-16,  5.66E-16};
    TGraph data_HMinus(35, x_HMinus, y_HMinus);

    TGraph data[4] = {data_HPlus, data_H2Plus, data_H3Plus, data_HMinus};

    //TCanvas tEnergyLossCanvas( "energyloss_canvas", "energy loss" );
    //TH1D tEnergyLossHisto("energy loss","energy loss",100,0,500);

    TCanvas tScatteringAngleCanvas("scatteringangle_canvas", "scattering_angle");
    TH1D tScatteringAngleHisto("scattering angle", "scattering angle", 100, -1., 1.);

    TCanvas tSecondaryEnergyCanvas("secondaryenergy_canvas", "secondary energy");
    TH1D tSecondaryEnergyHisto("secondary energy", "secondary energy", 100, 0, 500);

    double tLowEnergy = 1;
    double tHighEnergy = 100000;
    double Span = tHighEnergy - tLowEnergy;
    double nPoints = 1000;
    double tLength = 1.0;

    const char* particle_list[4] = {"H^+", "H_2^+", "H_3^+", "H^-"};  //103
    const char* particle_name[4] = {"H^{#plus}", "H_{2}^{#plus}", "H_{3}^{#plus}", "H^{#minus}"};
    //iterate over H+, H2+, H3+, H-
    for (int particle = 0; particle < 4; particle++) {
        // make particles
        KSParticle* tInitialParticle = KSParticleFactory::GetInstance().StringCreate(particle_list[particle]);
        KSParticle* tFinalParticle = KSParticleFactory::GetInstance().StringCreate(particle_list[particle]);
        KSParticleQueue tSecondaries;

        tInitialParticle->SetLength(0.00);
        tFinalParticle->SetLength(tLength);

        KThreeVector tDirection = KThreeVector(0., 0., 1.);

        double tEnergy = tLowEnergy;
        int tIndex = 0;
        while (tEnergy <= tHighEnergy) {
            tInitialParticle->SetMomentum(tDirection);
            tFinalParticle->SetMomentum(tDirection);

            tInitialParticle->SetKineticEnergy_eV(tEnergy);
            tFinalParticle->SetKineticEnergy_eV(tEnergy);

            double tAverageCrossSection = 0.;
            tScattering->CalculateAverageCrossSection(*tInitialParticle, *tFinalParticle, tAverageCrossSection);
            tCrossSectionGraphList[particle].SetPoint(tIndex,
                                                      tEnergy,
                                                      tAverageCrossSection * 1e4);  //Convert from m^2 to cm^2


            if (tAverageCrossSection > 0.) {
                tScattering->DiceCalculator(tAverageCrossSection);
                tSecondaries.clear();
                tScattering->ExecuteInteraction(*tInitialParticle, *tFinalParticle, tSecondaries);

                //tEnergyLossHisto.Fill( tInitialParticle->GetKineticEnergy_eV() - tFinalParticle->GetKineticEnergy_eV() );
                tScatteringAngleHisto.Fill(cos(tSecondaries[0]->GetPolarAngleToZ() * katrin::KConst::Pi() / 180));
                tSecondaryEnergyHisto.Fill(tSecondaries[0]->GetKineticEnergy_eV());
            }
            //Logarithmic step size in energy
            //Based on equation found on https://www.rohde-schwarz.com/us/faq/how-do-i-calculate-the-logarithmic-steps-faq_78704-30234.html (17 April 2017)
            tEnergy =
                pow(10, log10(Span) / nPoints * tIndex) + tLowEnergy;  //http://www.cplusplus.com/reference/cmath/log/
            tIndex++;
            mainmsg(eNormal) << "Step: " << tIndex << "     Energy: " << tEnergy << reom;
        }
    }

    // show plots
    tCrossSectionCanvas.cd(0);
    tCrossSectionCanvas.SetLogx();
    tCrossSectionCanvas.SetLogy();

    tCrossSectionGraphList[0].SetMinimum(1.e-21);
    tCrossSectionGraphList[0].SetMaximum(1.e-14);
    tCrossSectionGraphList[0].GetXaxis()->SetLimits(1, 1e6);
    tCrossSectionGraphList[0].SetTitle("Ionization cross section on H_{2}");
    tCrossSectionGraphList[0].GetXaxis()->SetTitle("Ion energy (eV)");
    tCrossSectionGraphList[0].GetXaxis()->SetTitleSize(0.06);
    tCrossSectionGraphList[0].GetXaxis()->SetTitleOffset(0.8);
    tCrossSectionGraphList[0].GetYaxis()->SetTitle("Cross section (cm^{2})");
    tCrossSectionGraphList[0].GetYaxis()->SetTitleSize(0.06);
    tCrossSectionGraphList[0].GetYaxis()->SetTitleOffset(0.8);

    EColor color_list[4] = {kRed, kGreen, kBlue, kOrange};
    for (int graph = 0; graph < 4; graph++) {
        tCrossSectionGraphList[graph].SetMarkerColor(color_list[graph]);
        tCrossSectionGraphList[graph].SetMarkerStyle(20);
        tCrossSectionGraphList[graph].SetMarkerSize(0.5);
        tCrossSectionGraphList[graph].SetLineWidth(1);
        tCrossSectionLegend.AddEntry(&tCrossSectionGraphList[graph], particle_name[graph], "p");
        if (graph == 0) {
            tCrossSectionGraphList[graph].Draw("AP");
        }
        else {
            tCrossSectionGraphList[graph].Draw("same p");
        }
    }

    //Cross section data points
    for (int graph = 0; graph < 4; graph++) {
        data[graph].SetMarkerStyle(4);
        data[graph].SetMarkerColor(color_list[graph] + 1);
        data[graph].Draw("same p");
    }

    TGraph data_blank;
    data_blank.SetMarkerColor(kBlack);
    data_blank.SetMarkerStyle(4);
    tCrossSectionLegend.AddEntry(&data_blank, "Data", "p");
    tCrossSectionLegend.Draw();

    /*tEnergyLossCanvas.SetLogy();
    tEnergyLossCanvas.cd( 0 );
    tEnergyLossHisto.SetMarkerColor( kRed );
    tEnergyLossHisto.SetMarkerStyle( 20 );
    tEnergyLossHisto.SetMarkerSize( 0.5 );
    tEnergyLossHisto.SetLineWidth( 1 );
    tEnergyLossHisto.SetTitle( "Ion energy loss" );
    tEnergyLossHisto.GetXaxis()->SetTitle( "Energy loss in eV" );
    tEnergyLossHisto.Draw();*/

    tScatteringAngleCanvas.cd(0);
    tScatteringAngleHisto.SetMinimum(0);
    tScatteringAngleHisto.SetMarkerColor(kRed);
    tScatteringAngleHisto.SetMarkerStyle(20);
    tScatteringAngleHisto.SetMarkerSize(0.5);
    tScatteringAngleHisto.SetLineWidth(1);
    tScatteringAngleHisto.SetTitle("Secondary electron scattering angle");
    tScatteringAngleHisto.GetXaxis()->SetTitle("cos(#theta)");
    tScatteringAngleHisto.Draw();

    tSecondaryEnergyCanvas.cd(0);
    tSecondaryEnergyCanvas.SetLogy();
    tSecondaryEnergyHisto.SetMarkerColor(kRed);
    tSecondaryEnergyHisto.SetMarkerStyle(20);
    tSecondaryEnergyHisto.SetMarkerSize(0.5);
    tSecondaryEnergyHisto.SetLineWidth(1);
    tSecondaryEnergyHisto.SetTitle("Electron energy");
    tSecondaryEnergyHisto.GetXaxis()->SetTitle("Electron energy (eV)");
    tSecondaryEnergyHisto.Draw();

    tApplication.Run();

    return 0;
}
