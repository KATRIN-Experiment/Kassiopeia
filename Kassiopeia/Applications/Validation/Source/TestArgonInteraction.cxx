#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KTagProcessor.hh"
#include "KElementProcessor.hh"

#include "KSToolbox.h"
#include "KSIntScattering.h"
#include "KSMainMessage.h"
#include "KSParticleFactory.h"

#include "KConst.h"

#include "TApplication.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TGraph.h"

#include "KSIntDensityConstant.h"
#include "KSIntCalculatorArgon.h"


#include <sstream>
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main( int anArgc, char** anArgv )
{
    double tLowEnergy = 1;
    double tHighEnergy = 18601;
    double tEnergyStepSize = 1.;
    double tLength = 1.0;

    KSIntDensityConstant* tDensityCalc = new KSIntDensityConstant();
    tDensityCalc->SetPressure(1.e-4); // pascal (*100 for mbar)
    tDensityCalc->SetTemperature(300.); // kelvin

    KSIntCalculator* elasticCalculator = new KSIntCalculatorArgonElastic();
    elasticCalculator->SetName( "argon_elastic" );
    elasticCalculator->SetTag( "argon" );

    KSIntCalculator* IonIntCalculator = new KSIntCalculatorArgonSingleIonisation();
    IonIntCalculator->SetName(  "argon_single_ionisation" );
    IonIntCalculator->SetTag( "argon" );

    KSIntCalculator* DoubleIonIntCalculator = new KSIntCalculatorArgonDoubleIonisation();
    DoubleIonIntCalculator->SetName(  "argon_double_ionisation" );
    DoubleIonIntCalculator->SetTag( "argon" );

    KSIntCalculator* ExcitationCalculators[25];
    for( unsigned int i = 0; i < 25; ++i )
    {
        stringstream tmp;
        tmp << (i + 1);
        ExcitationCalculators[i] = new KSIntCalculatorArgonExcitation();
        ExcitationCalculators[i]->SetName( "argon_excitation_state_" + tmp.str() );
        ExcitationCalculators[i]->SetTag( "argon" );
        static_cast< KSIntCalculatorArgonExcitation* >( ExcitationCalculators[i] )->SetExcitationState( i + 1 );
    }


    // get stuff from toolbox == fuck the toolbox
    KSIntScattering* tScattering = new KSIntScattering();
    tScattering->SetDensity( tDensityCalc );
    tScattering->SetSplit( false );

    tScattering->AddCalculator(elasticCalculator);
    tScattering->AddCalculator(IonIntCalculator);
    tScattering->AddCalculator(DoubleIonIntCalculator);
    for( unsigned int i = 0; i < 25; ++i )
    {
        tScattering->AddCalculator(ExcitationCalculators[i]);
    }


    tScattering->Initialize();

    // initialize root
    TApplication tApplication( "Test Interaction", 0, NULL );

    TCanvas tCrossSectionCanvas( "crosssection_canvas", "cross section" );
    TGraph tCrossSectionGraph;
    TGraph tIonCrossSectionGraph;
    TGraph tDoubleIonCrossSectionGraph;
    TGraph tElCrossSectionGraph;
    TGraph tExcitationGraphs[25];


    TCanvas tEnergyLossCanvas( "energyloss_canvas", "energy loss" );
    TH1D tEnergyLossHisto("energy loss","energy loss",400,7.,120.);

    TCanvas tScatteringAngleCanvas( "scatteringangle_canvas", "scattering_angle" );
    TH1D tScatteringAngleHisto("scattering angle","scattering angle",200,0.,180.);


    // make particles
    KSParticle* tInitialParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticle* tFinalParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticle* tInteractionParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticleQueue tSecondaries;

    tInitialParticle->SetLength( 0.00 );
    tFinalParticle->SetLength( tLength );


    double tEnergy = tLowEnergy;
    int tIndex = 0;    
    while ( tEnergy <= tHighEnergy )
    {
		tInitialParticle->SetKineticEnergy_eV( tEnergy );
        tInteractionParticle->SetKineticEnergy_eV( tEnergy );
		tFinalParticle->SetKineticEnergy_eV( tEnergy );


        double tAverageCrossSection = 0.;
        tScattering->CalculateAverageCrossSection( *tInitialParticle, *tFinalParticle, tAverageCrossSection);
        tCrossSectionGraph.SetPoint ( tIndex, tEnergy, tAverageCrossSection );


        double tCrossSection = 0.;

        IonIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tIonCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);

        DoubleIonIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tDoubleIonCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);

        elasticCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tElCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);

        for( unsigned int i = 0; i < 25; ++i )
        {
            ExcitationCalculators[i]->CalculateCrossSection(*tInitialParticle, tCrossSection);
            tExcitationGraphs[i].SetPoint(tIndex, tEnergy, tCrossSection);
        }

        tEnergy += tEnergyStepSize;
        tIndex++;

        mainmsg(eNormal) << "Calculating Crossections - Step: " << tIndex << "     Energy: " << tEnergy << reom;
    }

    tEnergy = 500.;
    KThreeVector tDirection = KThreeVector(0.,0.,1.);
    tInitialParticle->SetMomentum(tDirection);
    tInteractionParticle->SetMomentum(tDirection);
    tFinalParticle->SetMomentum(tDirection);
    tInitialParticle->SetKineticEnergy_eV( tEnergy );
    tInteractionParticle->SetKineticEnergy_eV( tEnergy );
    tFinalParticle->SetKineticEnergy_eV( tEnergy );

    for(int i = 0; i < 100000; i++)
    {
        mainmsg(eNormal) << "Calculating Energylossfunction - Step: " << i << "     Energy: " << tEnergy << reom;
        double tAverageCrossSection = 0.;
        tScattering->CalculateAverageCrossSection( *tInitialParticle, *tFinalParticle, tAverageCrossSection);
        if(tAverageCrossSection > 0.)
        {
            tScattering->DiceCalculator( tAverageCrossSection );
            tScattering->ExecuteInteraction( *tInteractionParticle, *tFinalParticle, tSecondaries);
            tEnergyLossHisto.Fill( tInteractionParticle->GetKineticEnergy_eV() - tFinalParticle->GetKineticEnergy_eV() );
            tScatteringAngleHisto.Fill( tFinalParticle->GetPolarAngleToZ() );
            tFinalParticle->SetMomentum(tDirection);
            tFinalParticle->SetKineticEnergy_eV(tEnergy);
        }
    }
    

    // show plots

    tCrossSectionCanvas.cd( 0 );
    tCrossSectionCanvas.SetLogx();
    tCrossSectionCanvas.SetLogy();
    tCrossSectionGraph.SetMarkerColor( kRed );
    tCrossSectionGraph.SetMarkerStyle( 20 );
    tCrossSectionGraph.SetMarkerSize( 0.5 );
    tCrossSectionGraph.SetLineWidth( 1 );
    tCrossSectionGraph.SetMinimum(1.e-24);
    tCrossSectionGraph.SetMaximum(1.e-17);
    tCrossSectionGraph.SetTitle( "cross section" );
    tCrossSectionGraph.GetXaxis()->SetTitle( "Energy in eV" );    
    tCrossSectionGraph.GetYaxis()->SetTitle( "cross section in m^2" );
    tCrossSectionGraph.GetXaxis()->SetRangeUser(1.,40000.);
    tCrossSectionGraph.Draw( "AP" );

    tIonCrossSectionGraph.SetMarkerColor( kGreen );
    tIonCrossSectionGraph.SetMarkerStyle( 20 );
    tIonCrossSectionGraph.SetMarkerSize( 0.5 );
    tIonCrossSectionGraph.SetLineWidth( 1 );

    tIonCrossSectionGraph.Draw("same");

    tDoubleIonCrossSectionGraph.SetMarkerColor( kGreen );
    tDoubleIonCrossSectionGraph.SetMarkerStyle( 20 );
    tDoubleIonCrossSectionGraph.SetMarkerSize( 0.5 );
    tDoubleIonCrossSectionGraph.SetLineWidth( 1 );
    tDoubleIonCrossSectionGraph.Draw("same");

    tElCrossSectionGraph.SetMarkerColor( kBlue );
    tElCrossSectionGraph.SetMarkerStyle( 20 );
    tElCrossSectionGraph.SetMarkerSize( 0.5 );
    tElCrossSectionGraph.SetLineWidth( 1 );
    tElCrossSectionGraph.Draw("same");

    for( unsigned int i = 0; i < 25; ++i )
    {
        tExcitationGraphs[i].SetMarkerColor(kYellow);
        tExcitationGraphs[i].SetMarkerStyle(20);
        tExcitationGraphs[i].SetMarkerSize( 0.5 );
        tExcitationGraphs[i].SetLineWidth( 1 );
        tExcitationGraphs[i].Draw("same");
    }


    tEnergyLossCanvas.cd( 0 );
    tEnergyLossHisto.SetMarkerColor( kRed );
    tEnergyLossHisto.SetMarkerStyle( 20 );
    tEnergyLossHisto.SetMarkerSize( 0.5 );
    tEnergyLossHisto.SetLineWidth( 1 );
    tEnergyLossHisto.SetTitle( "energy loss" );
    tEnergyLossHisto.GetXaxis()->SetTitle( "Energy loss in eV" );
    tEnergyLossHisto.Draw();

    tScatteringAngleCanvas.cd( 0 );
    tScatteringAngleHisto.SetMarkerColor( kRed );
    tScatteringAngleHisto.SetMarkerStyle( 20 );
    tScatteringAngleHisto.SetMarkerSize( 0.5 );
    tScatteringAngleHisto.SetLineWidth( 1 );
    tScatteringAngleHisto.SetTitle( "scattering angle" );
    tScatteringAngleHisto.GetXaxis()->SetTitle( "scattering angle in degree" );
    tScatteringAngleHisto.Draw();


    tApplication.Run();

    // deinitialize kassiopeia

    KSToolbox::DeleteInstance();

    return 0;
}

