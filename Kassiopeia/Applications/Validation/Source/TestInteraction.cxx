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

    KTextFile tInputFile;
    tInputFile.AddToBases( "TestSpaceInteraction.xml" );
    tInputFile.AddToPaths( string( CONFIG_DEFAULT_DIR ) + string( "/Validation" ) );
    tXMLTokenizer.ProcessFile( &tInputFile );

    // read in command line parameters
    if( anArgc != 2 )
    {
        mainmsg( eWarning ) << "usage:" << ret;
        mainmsg << "TestInteraction <KSINTSCATTERINGNAME>"<< eom;
        return -1;
    }

    string tScatteringName = anArgv[1];
    mainmsg( eNormal ) <<"Using Scattering with name <"<<tScatteringName<<">"<<eom;

    double tLowEnergy = 200.0;
    double tHighEnergy = 12000.0;
    double tEnergyStepSize = 0.2;
    double tLength = 1.0;
    double tDensity = 100 * 1.0e-2 / (KConst::kB() * 300.0 ); //take value from density calculator in xml file


    // get stuff from toolbox
    KSIntScattering* tScattering = KSToolbox::GetInstance()->GetObjectAs<KSIntScattering>( tScatteringName );
    tScattering->Initialize();

    // initialize root
    TApplication tApplication( "Test Interaction", 0, NULL );

    TCanvas tProbabilityCanvas( "probability_canvas", "interaction probability" );
    TGraph tInteractionProbabilityGraph;

    TCanvas tCrossSectionCanvas( "crosssection_canvas", "cross section" );
    TGraph tCrossSectionGraph;

    TCanvas tEnergyLossCanvas( "energyloss_canvas", "energy loss" );
    TH1D tEnergyLossHisto("energy loss","energy loss",5000,0.,200.);

//    TCanvas tScatteringNameCanvas( "scattering_name_canvas", "scattering name" );
//    TH1D tScatteringNameHisto;


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

        double tCrossSection;
        tScattering->CalculateAverageCrossSection( *tInitialParticle, *tFinalParticle, tCrossSection);
//		double tCrossSection = log ( 1.0 - tProbability ) / ( -1.0 * tLength * tDensity );
        tCrossSectionGraph.SetPoint ( tIndex, tEnergy, tCrossSection );

        double tProbability;
        tProbability = 1.0 - exp(-1.* tCrossSection * tLength * tDensity);
//		tScattering->CalculateInteractionProbability( *tInitialParticle, *tFinalParticle, tProbability );
        tInteractionProbabilityGraph.SetPoint( tIndex, tEnergy, tProbability );

        //tScattering->ExecuteInteraction( *tInitialParticle, *tFinalParticle, *tInteractionParticle, tSecondaries);       
        if(tCrossSection > 0.)
        {
//            tScattering->DiceCalculator( tCrossSection );
            tScattering->ExecuteInteraction( *tInteractionParticle, *tFinalParticle, tSecondaries);
            tEnergyLossHisto.Fill( tInteractionParticle->GetKineticEnergy_eV() - tFinalParticle->GetKineticEnergy_eV() );
        }        
        tEnergy += tEnergyStepSize;
		tIndex++;

        mainmsg(eNormal) << "Step: " << tIndex << "     Energy: " << tEnergy << reom;
    }



    // show plots
    tProbabilityCanvas.cd( 0 );
    tInteractionProbabilityGraph.SetMarkerColor( kRed );
    tInteractionProbabilityGraph.SetMarkerStyle( 20 );
    tInteractionProbabilityGraph.SetMarkerSize( 0.5 );
    tInteractionProbabilityGraph.SetLineWidth( 1 );
    tInteractionProbabilityGraph.SetTitle( "Interaction Probability" );
    tInteractionProbabilityGraph.GetXaxis()->SetTitle( "Energy in eV" );
    tInteractionProbabilityGraph.GetYaxis()->SetTitle( "Interaction probability" );
    tInteractionProbabilityGraph.Draw( "AP" );

    tCrossSectionCanvas.cd( 0 );
    tCrossSectionGraph.SetMarkerColor( kRed );
    tCrossSectionGraph.SetMarkerStyle( 20 );
    tCrossSectionGraph.SetMarkerSize( 0.5 );
    tCrossSectionGraph.SetLineWidth( 1 );
    tCrossSectionGraph.SetTitle( "cross section" );
    tCrossSectionGraph.GetXaxis()->SetTitle( "Energy in eV" );
    tCrossSectionGraph.GetYaxis()->SetTitle( "cross section in 1/m��" );
    tCrossSectionGraph.Draw( "AP" );

    tEnergyLossCanvas.cd( 0 );
    tEnergyLossHisto.SetMarkerColor( kRed );
    tEnergyLossHisto.SetMarkerStyle( 20 );
    tEnergyLossHisto.SetMarkerSize( 0.5 );
    tEnergyLossHisto.SetLineWidth( 1 );
    tEnergyLossHisto.SetTitle( "energy loss" );
    tEnergyLossHisto.GetXaxis()->SetTitle( "Energy loss in eV" );
    tEnergyLossHisto.DrawNormalized();


    tApplication.Run();

    // deinitialize kassiopeia

    KSToolbox::DeleteInstance();

    return 0;
}

