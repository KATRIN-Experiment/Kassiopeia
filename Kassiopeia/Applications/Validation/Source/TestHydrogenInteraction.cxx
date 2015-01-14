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
#include "KSIntCalculatorHydrogen.h"


#include <sstream>
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main( int anArgc, char** anArgv )
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

    tVariableProcessor.InsertAfter( &tXMLTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tIncludeProcessor.InsertAfter( &tFormulaProcessor );
    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tTagProcessor.InsertAfter( &tConditionProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    double tLowEnergy = 1;
    double tHighEnergy = 18601;
    double tEnergyStepSize = .1;
    double tLength = 1.0;

    KSIntDensityConstant* tDensityCalc = new KSIntDensityConstant();
    tDensityCalc->SetPressure(1.e-4); // pascal (*100 for mbar)
    tDensityCalc->SetTemperature(300.); // kelvin

    KSIntCalculator* elasticCalculator = new KSIntCalculatorHydrogenElastic();
    elasticCalculator->SetName( "hydrogen_elastic" );
    elasticCalculator->SetTag( "hydrogen" );

    KSIntCalculator* vibCalculator = new KSIntCalculatorHydrogenVib();
    vibCalculator->SetName( "hydrogen_vib" );
    vibCalculator->SetTag( "hydrogen" );

    KSIntCalculator* rot02Calculator = new KSIntCalculatorHydrogenRot02();
    rot02Calculator->SetName(  "hydrogen_rot02" );
    rot02Calculator->SetTag( "hydrogen" );

    KSIntCalculator* rot13Calculator = new KSIntCalculatorHydrogenRot13();
    rot13Calculator->SetName( "hydrogen_rot13" );
    rot13Calculator->SetTag( "hydrogen" );

    KSIntCalculator* rot20Calculator = new KSIntCalculatorHydrogenRot20();
    rot20Calculator->SetName(  "hydrogen_rot20" );
    rot20Calculator->SetTag( "hydrogen" );

    KSIntCalculator* excBIntCalculator = new KSIntCalculatorHydrogenExcitationB();
    excBIntCalculator->SetName( "hydrogen_excB" );
    excBIntCalculator->SetTag( "hydrogen" );

    KSIntCalculator* excCIntCalculator = new KSIntCalculatorHydrogenExcitationC();
    excCIntCalculator->SetName(  "hydrogen_excC" );
    excCIntCalculator->SetTag( "hydrogen" );

    KSIntCalculator* diss10IntCalculator = new KSIntCalculatorHydrogenDissoziation10();
    diss10IntCalculator->SetName( "hydrogen_diss10" );
    diss10IntCalculator->SetTag( "hydrogen" );

    KSIntCalculator* diss15IntCalculator = new KSIntCalculatorHydrogenDissoziation15();
    diss15IntCalculator->SetName( "hydrogen_diss15" );
    diss15IntCalculator->SetTag( "hydrogen" );

    KSIntCalculator* excElIntCalculator = new KSIntCalculatorHydrogenExcitationElectronic();
    excElIntCalculator->SetName( "hydrogen_excEl");
    excElIntCalculator->SetTag( "hydrogen" );

    KSIntCalculator* IonIntCalculator = new KSIntCalculatorHydrogenIonisation();
    IonIntCalculator->SetName(  "hydrogen_ionisation" );
    IonIntCalculator->SetTag( "hydrogen" );


    // get stuff from toolbox == fuck the toolbox
    KSIntScattering* tScattering = new KSIntScattering();
    tScattering->SetDensity( tDensityCalc );
    tScattering->SetSplit( false );

//    tScattering->AddCalculator(elasticCalculator);
    tScattering->AddCalculator(vibCalculator);
    tScattering->AddCalculator(rot02Calculator);
    tScattering->AddCalculator(rot13Calculator);
    tScattering->AddCalculator(rot20Calculator);
    tScattering->AddCalculator(excBIntCalculator);
    tScattering->AddCalculator(excCIntCalculator);
    tScattering->AddCalculator(diss10IntCalculator);
    tScattering->AddCalculator(diss15IntCalculator);
    tScattering->AddCalculator(excElIntCalculator);
    tScattering->AddCalculator(IonIntCalculator);


    tScattering->Initialize();

    // initialize root
    TApplication tApplication( "Test Interaction", 0, NULL );

    TCanvas tCrossSectionCanvas( "crosssection_canvas", "cross section" );
    TGraph tCrossSectionGraph;
    TGraph tIonCrossSectionGraph;
    TGraph tElCrossSectionGraph;
    TGraph tVibCrossSectionGraph;
    TGraph trot02CrossSectionGraph;
    TGraph trot13CrossSectionGraph;
    TGraph trot20CrossSectionGraph;
    TGraph texcBCrossSectionGraph;
    TGraph texcCCrossSectionGraph;
    TGraph tdiss10CrossSectionGraph;
    TGraph tdiss15rossSectionGraph;
    TGraph texcElCrossSectionGraph;

    TCanvas tEnergyLossCanvas( "energyloss_canvas", "energy loss" );
    TH1D tEnergyLossHisto("energy loss","energy loss",1000,-0.2,100);

    TCanvas tScatteringAngleCanvas( "scatteringangle_canvas", "scattering_angle" );
    TH1D tScatteringAngleHisto("scattering angle","scattering angle",100,0.,180.);


    // make particles
    KSParticle* tInitialParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticle* tFinalParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticle* tInteractionParticle = KSParticleFactory::GetInstance()->Create( 11 );
    KSParticleQueue tSecondaries;

    tInitialParticle->SetLength( 0.00 );
    tFinalParticle->SetLength( tLength );

    KThreeVector tDirection = KThreeVector(0.,0.,1.);

    double tEnergy = tLowEnergy;
    int tIndex = 0;    
    while ( tEnergy <= tHighEnergy )
    {
        tInitialParticle->SetMomentum(tDirection);
        tInteractionParticle->SetMomentum(tDirection);
        tFinalParticle->SetMomentum(tDirection);
		tInitialParticle->SetKineticEnergy_eV( tEnergy );
        tInteractionParticle->SetKineticEnergy_eV( tEnergy );
		tFinalParticle->SetKineticEnergy_eV( tEnergy );

        double tAverageCrossSection = 0.;
        tScattering->CalculateAverageCrossSection( *tInitialParticle, *tFinalParticle, tAverageCrossSection);
//        std::cout << "total cross section: " << tAverageCrossSection << std::endl;
        tCrossSectionGraph.SetPoint ( tIndex, tEnergy, tAverageCrossSection );
        double tCrossSection = 0.;
        IonIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tIonCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "ion cross section " << tCrossSection << std::endl;
//        elasticCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
//        tElCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
        vibCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tVibCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
        rot02Calculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        trot02CrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
        rot13Calculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        trot13CrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
        rot20Calculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        trot20CrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
        excBIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        texcBCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "excB cross section " << tCrossSection << std::endl;
        excCIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        texcCCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "excC cross section " << tCrossSection << std::endl;
        diss10IntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tdiss10CrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "diss10 cross section " << tCrossSection << std::endl;
        diss15IntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        tdiss15rossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "diss15 cross section " << tCrossSection << std::endl;
        excElIntCalculator->CalculateCrossSection(*tInitialParticle, tCrossSection);
        texcElCrossSectionGraph.SetPoint( tIndex, tEnergy, tCrossSection);
//        std::cout << "excEl cross section " << tCrossSection << std::endl;

        if(tAverageCrossSection > 0.)
        {
            tScattering->DiceCalculator( tAverageCrossSection );
            tScattering->ExecuteInteraction( *tInteractionParticle, *tFinalParticle, tSecondaries);
            tEnergyLossHisto.Fill( tInteractionParticle->GetKineticEnergy_eV() - tFinalParticle->GetKineticEnergy_eV() );
            tScatteringAngleHisto.Fill( tFinalParticle->GetPolarAngleToZ() );
        }
        tEnergy += tEnergyStepSize;
        tIndex++;

        mainmsg(eNormal) << "Step: " << tIndex << "     Energy: " << tEnergy << reom;
    }

    // show plots

    tCrossSectionCanvas.cd( 0 );
    tCrossSectionGraph.SetMarkerColor( kRed );
    tCrossSectionGraph.SetMarkerStyle( 20 );
    tCrossSectionGraph.SetMarkerSize( 0.5 );
    tCrossSectionGraph.SetLineWidth( 1 );
    tCrossSectionGraph.SetTitle( "cross section" );
    tCrossSectionGraph.GetXaxis()->SetTitle( "Energy in eV" );
    tCrossSectionGraph.GetYaxis()->SetTitle( "cross section in 1/m^2" );
    tCrossSectionGraph.Draw( "AP" );

    tIonCrossSectionGraph.SetMarkerColor( kGreen );
    tIonCrossSectionGraph.SetMarkerStyle( 20 );
    tIonCrossSectionGraph.SetMarkerSize( 0.5 );
    tIonCrossSectionGraph.SetLineWidth( 1 );
    tIonCrossSectionGraph.Draw("same");

//    tElCrossSectionGraph.SetMarkerColor( kBlue );
//    tElCrossSectionGraph.SetMarkerStyle( 20 );
//    tElCrossSectionGraph.SetMarkerSize( 0.5 );
//    tElCrossSectionGraph.SetLineWidth( 1 );
//    tElCrossSectionGraph.Draw("same");

    tVibCrossSectionGraph.SetMarkerColor( kYellow );
    tVibCrossSectionGraph.SetMarkerStyle( 20 );
    tVibCrossSectionGraph.SetMarkerSize( 0.5 );
    tVibCrossSectionGraph.SetLineWidth( 1 );
    tVibCrossSectionGraph.Draw("same");

    trot02CrossSectionGraph.SetMarkerColor( kGreen );
    trot02CrossSectionGraph.SetMarkerStyle( 20 );
    trot02CrossSectionGraph.SetMarkerSize( 0.5 );
    trot02CrossSectionGraph.SetLineWidth( 1 );
    trot02CrossSectionGraph.Draw("same");

    trot13CrossSectionGraph.SetMarkerColor( kGreen );
    trot13CrossSectionGraph.SetMarkerStyle( 20 );
    trot13CrossSectionGraph.SetMarkerSize( 0.5 );
    trot13CrossSectionGraph.SetLineWidth( 1 );
    trot13CrossSectionGraph.Draw("same");

    trot20CrossSectionGraph.SetMarkerColor( kYellow );
    trot20CrossSectionGraph.SetMarkerStyle( 20 );
    trot20CrossSectionGraph.SetMarkerSize( 0.5 );
    trot20CrossSectionGraph.SetLineWidth( 1 );
    trot20CrossSectionGraph.Draw("same");

    texcBCrossSectionGraph.SetMarkerColor( kYellow );
    texcBCrossSectionGraph.SetMarkerStyle( 20 );
    texcBCrossSectionGraph.SetMarkerSize( 0.5 );
    texcBCrossSectionGraph.SetLineWidth( 1 );
    texcBCrossSectionGraph.Draw("same");
    texcBCrossSectionGraph.SetMarkerColor( kYellow );

    texcCCrossSectionGraph.SetMarkerColor( kRed );
    texcCCrossSectionGraph.SetMarkerStyle( 20 );
    texcCCrossSectionGraph.SetMarkerSize( 0.5 );
    texcCCrossSectionGraph.SetLineWidth( 1 );    
    texcCCrossSectionGraph.Draw("same");

    tdiss10CrossSectionGraph.SetMarkerColor( kYellow );
    tdiss10CrossSectionGraph.SetMarkerStyle( 20 );
    tdiss10CrossSectionGraph.SetMarkerSize( 0.5 );
    tdiss10CrossSectionGraph.SetLineWidth( 1 );
    tdiss10CrossSectionGraph.Draw("same");

    tdiss15rossSectionGraph.SetMarkerColor( kYellow );
    tdiss15rossSectionGraph.SetMarkerStyle( 20 );
    tdiss15rossSectionGraph.SetMarkerSize( 0.5 );
    tdiss15rossSectionGraph.SetLineWidth( 1 );
    tdiss15rossSectionGraph.Draw("same");

    texcElCrossSectionGraph.SetMarkerColor( kYellow );
    texcElCrossSectionGraph.SetMarkerStyle( 20 );
    texcElCrossSectionGraph.SetMarkerSize( 0.5 );
    texcElCrossSectionGraph.SetLineWidth( 1 );
    texcElCrossSectionGraph.Draw("same");

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

