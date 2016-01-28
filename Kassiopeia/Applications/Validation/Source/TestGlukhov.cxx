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
#include "KSIntDecay.h"
#include "KSMainMessage.h"
#include "KSParticleFactory.h"

#include "KConst.h"

#include "TApplication.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TGraph.h"

#include "KSIntDecayCalculatorGlukhovDeExcitation.h"
#include "KSIntDecayCalculatorGlukhovExcitation.h"
#include "KSIntDecayCalculatorGlukhovIonisation.h"
#include "KSIntDecayCalculatorGlukhovSpontaneous.h"

#include <sstream>
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main( int /*anArgc*/, char** /*anArgv*/ )
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

    //double tLowEnergy = 1;
    //double tHighEnergy = 18601;
    //double tEnergyStepSize = .1;
    double tLength = 1.0;

    KSIntDecayCalculatorGlukhovSpontaneous* tSpontaneus = new KSIntDecayCalculatorGlukhovSpontaneous();
    tSpontaneus->SetName("spon");
    tSpontaneus->SetTag("spon");
    tSpontaneus->SetTargetPID(10000);

    KSIntDecayCalculatorGlukhovIonisation* tIon = new KSIntDecayCalculatorGlukhovIonisation();
    tIon->SetName("ion");
    tIon->SetTag("ion");
    tIon->SetTargetPID(10000);
    tIon->SetTemperature(300.);

    KSIntDecayCalculatorGlukhovExcitation* tExc = new KSIntDecayCalculatorGlukhovExcitation();
    tExc->SetName("exc");
    tExc->SetTag("exc");
    tExc->SetTargetPID(10000);
    tExc->SetTemperature(300.);

    KSIntDecayCalculatorGlukhovDeExcitation* tDex = new KSIntDecayCalculatorGlukhovDeExcitation();
    tDex->SetName("dex");
    tDex->SetTag("dex");
    tDex->SetTargetPID(10000);
    tDex->SetTemperature(300.);



    KSIntDecay* tDecay = new KSIntDecay();
    tDecay->SetSplit( false );
    tDecay->AddCalculator(tSpontaneus);
    tDecay->AddCalculator(tIon);
    tDecay->AddCalculator(tExc);
    tDecay->AddCalculator(tDex);

    tDecay->Initialize();

    // initialize root
    TApplication tApplication( "Test Glukhov", 0, NULL );

    TCanvas tLifeTimeCanvas( "lifetime_canvas", "lifetime" );
    TGraph tLifeTimeGraph;
    TGraph tIonLifeTimeGraph;
    TGraph tExclLifeTimeGraph;
    TGraph tDexlLifeTimeGraph;
    TGraph tSponlLifeTimeGraph;



    // make particles
    KSParticle* tInitialParticle = KSParticleFactory::GetInstance()->Create( 10000 );
    KSParticle* tFinalParticle = KSParticleFactory::GetInstance()->Create( 10000 );
    //KSParticle* tInteractionParticle = KSParticleFactory::GetInstance()->Create( 10000 );
    KSParticleQueue tSecondaries;

    tInitialParticle->SetLength( 0.00 );
    tFinalParticle->SetLength( tLength );

    KThreeVector tDirection = KThreeVector(0.,0.,1.);
    int l = 0;
    int tIndex = 0;
    for(int n = 10; n < 1000; n++)
    {
        tInitialParticle->SetMainQuantumNumber(n);
        tInitialParticle->SetSecondQuantumNumber(l);
        tFinalParticle->SetMainQuantumNumber(n);
        tFinalParticle->SetSecondQuantumNumber(l);


        double tTotalRate = 0.;
        double tLifetime = 0.;

        tIon->CalculateLifeTime(*tInitialParticle,tLifetime);
        tTotalRate += 1./tLifetime;
        tIonLifeTimeGraph.SetPoint( tIndex, n, tLifetime);
        tExc->CalculateLifeTime(*tInitialParticle,tLifetime);
        tTotalRate += 1./tLifetime;
        tExclLifeTimeGraph.SetPoint(tIndex,n,tLifetime);
        tDex->CalculateLifeTime(*tInitialParticle,tLifetime);
        tTotalRate += 1./tLifetime;
        tDexlLifeTimeGraph.SetPoint(tIndex,n,tLifetime);
        tSpontaneus->CalculateLifeTime(*tInitialParticle,tLifetime);
        tTotalRate += 1./tLifetime;
        tSponlLifeTimeGraph.SetPoint(tIndex,n,tLifetime);
        tLifeTimeGraph.SetPoint ( tIndex, n, 1./tTotalRate );



//        if(tAverageCrossSection > 0.)
//        {
//            tScattering->DiceCalculator( tAverageCrossSection );
//            tScattering->ExecuteInteraction( *tInteractionParticle, *tFinalParticle, tSecondaries);
//            tEnergyLossHisto.Fill( tInteractionParticle->GetKineticEnergy_eV() - tFinalParticle->GetKineticEnergy_eV() );
//            tScatteringAngleHisto.Fill( tFinalParticle->GetPolarAngleToZ() );
//        }
//        tEnergy += tEnergyStepSize;
        tIndex++;

        mainmsg(eNormal) << "Step: " << tIndex << "     n: " << n << reom;
    }

//    int n = 900;
//    l = 0;
//    tInitialParticle->SetMainQuantumNumber(n);
//    tInitialParticle->SetSecondQuantumNumber(l);
//    tFinalParticle->SetMainQuantumNumber(n);
//    tFinalParticle->SetSecondQuantumNumber(l);
//    tInteractionParticle->SetMainQuantumNumber(n);
//    tInteractionParticle->SetSecondQuantumNumber(l);

//    double tAverageLifetime = 0.;
//    tDecay->CalculateAverageLifetime(*tInitialParticle,*tFinalParticle,tAverageLifetime);

//    for(int i=0;i<10000;i++)
//    {
//        tDecay->DiceCalculator(tAverageLifetime);
//        tDecay->ExecuteInteraction(*tInteractionParticle,tFinalParticle,tSecondaries);
//        if(tFinalParticle->IsActive())

//    }

    // show plots

    tLifeTimeCanvas.cd( 0 );
    tLifeTimeCanvas.SetLogx();
    tLifeTimeCanvas.SetLogy();

    tLifeTimeGraph.SetMarkerColor( kBlue );
    tLifeTimeGraph.SetMarkerStyle( 20 );
    tLifeTimeGraph.SetMarkerSize( 0.5 );
    tLifeTimeGraph.SetLineWidth( 1);
    tLifeTimeGraph.SetTitle( "cross section" );
    tLifeTimeGraph.GetXaxis()->SetTitle( "Energy in eV" );
    tLifeTimeGraph.GetYaxis()->SetTitle( "cross section in 1/m^2" );
    tLifeTimeGraph.Draw( "AP" );
    tLifeTimeGraph.GetXaxis()->SetRangeUser(10.,1000.);
    tLifeTimeGraph.GetYaxis()->SetRangeUser(1.e-6,1.);

    tIonLifeTimeGraph.SetMarkerColor( kYellow );
    tIonLifeTimeGraph.SetMarkerStyle( 20 );
    tIonLifeTimeGraph.SetMarkerSize( 0.5 );
    tIonLifeTimeGraph.SetLineWidth( 1 );
    tIonLifeTimeGraph.Draw("same");
    tIonLifeTimeGraph.GetXaxis()->SetRangeUser(10.,1000.);
    tIonLifeTimeGraph.GetYaxis()->SetRangeUser(1.e-6,1.);

    tExclLifeTimeGraph.SetMarkerColor( kGreen );
    tExclLifeTimeGraph.SetMarkerStyle( 20 );
    tExclLifeTimeGraph.SetMarkerSize( 0.5 );
    tExclLifeTimeGraph.SetLineWidth( 1 );
    tExclLifeTimeGraph.Draw("same");
    tExclLifeTimeGraph.GetXaxis()->SetRangeUser(10.,1000.);
    tExclLifeTimeGraph.GetYaxis()->SetRangeUser(1.e-6,1.);

    tDexlLifeTimeGraph.SetMarkerColor( kGreen );
    tDexlLifeTimeGraph.SetMarkerStyle( 20 );
    tDexlLifeTimeGraph.SetMarkerSize( 0.5 );
    tDexlLifeTimeGraph.SetLineWidth( 1 );
    tDexlLifeTimeGraph.Draw("same");
    tDexlLifeTimeGraph.GetXaxis()->SetRangeUser(10.,1000.);
    tDexlLifeTimeGraph.GetYaxis()->SetRangeUser(1.e-6,1.);

    tSponlLifeTimeGraph.SetMarkerColor( kYellow );
    tSponlLifeTimeGraph.SetMarkerStyle( 20 );
    tSponlLifeTimeGraph.SetMarkerSize( 0.5 );
    tSponlLifeTimeGraph.SetLineWidth( 1 );
    tSponlLifeTimeGraph.Draw("same");
    tSponlLifeTimeGraph.GetXaxis()->SetRangeUser(10.,1000.);
    tSponlLifeTimeGraph.GetYaxis()->SetRangeUser(1.e-6,1.);

    tLifeTimeCanvas.Update();
    tApplication.Run();

    // deinitialize kassiopeia

    KSToolbox::DeleteInstance();

    return 0;
}

