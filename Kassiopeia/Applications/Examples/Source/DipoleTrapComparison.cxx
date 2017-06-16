#include "KSReadFileROOT.h"
#include "TGraph.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TAxis.h"

#include <vector>

using namespace Kassiopeia;
using namespace std;

int main()
{

    KRootFile* exact_tRootFile = new KRootFile();
    exact_tRootFile->AddToNames( "~/Work/kasper/install/output/Kassiopeia/DipoleNeutronTrapSimulation.root" );

    KSReadFileROOT exact_tReader;
    exact_tReader.OpenFile(exact_tRootFile );

    KSReadRunROOT& exact_tRunReader = exact_tReader.GetRun();
    KSReadEventROOT& exact_tEventReader = exact_tReader.GetEvent();
    KSReadTrackROOT& exact_tTrackReader = exact_tReader.GetTrack();
    KSReadStepROOT& exact_tStepReader= exact_tReader.GetStep();

    KSReadObjectROOT& exact_tCell = exact_tStepReader.GetObject( "component_step_world" );
    KSDouble& exact_tTime = exact_tCell.Get< KSDouble >( "time" );
    KSThreeVector& exact_tMomentum = exact_tCell.Get< KSThreeVector >( "momentum" );

    vector<Double_t>* exact_time_list = new vector<Double_t>();
    vector<Double_t>* exact_momentum_z_list = new vector<Double_t>();
    int i = 0;

    for( exact_tRunReader = 0; exact_tRunReader <= exact_tRunReader.GetLastRunIndex(); exact_tRunReader++ )
    {
  		for( exact_tEventReader = exact_tRunReader.GetFirstEventIndex(); exact_tEventReader <= exact_tRunReader.GetLastEventIndex(); exact_tEventReader++ )
  		{
  			for( exact_tTrackReader = exact_tEventReader.GetFirstTrackIndex(); exact_tTrackReader <= exact_tEventReader.GetLastTrackIndex(); exact_tTrackReader++ )
  			{
  				for( exact_tStepReader = exact_tTrackReader.GetFirstStepIndex(); exact_tStepReader <= exact_tTrackReader.GetLastStepIndex(); exact_tStepReader++ )
  				{
  					if( exact_tCell.Valid() && i % 1000 == 999)
  					{
              //std::cout << "Time: " << exact_tTime.Value() << "\t\tMomentum_Z: " << exact_tMomentum.Value().Z() << "\n";
              exact_time_list->push_back( exact_tTime.Value() );
              exact_momentum_z_list->push_back( exact_tMomentum.Value().Z() );
            }
            i++;
  				}
  			}
  		}
    }

    //exact_tReader.CloseFile();

    delete exact_tRootFile;

    KRootFile* adiabatic_tRootFile = new KRootFile();
    adiabatic_tRootFile->AddToNames( "~/Work/kasper/install/output/Kassiopeia/DipoleNeutronAdiabaticTrapSimulation.root" );

    KSReadFileROOT adiabatic_tReader;
    adiabatic_tReader.OpenFile( adiabatic_tRootFile );

    KSReadRunROOT& adiabatic_tRunReader = adiabatic_tReader.GetRun();
    KSReadEventROOT& adiabatic_tEventReader = adiabatic_tReader.GetEvent();
    KSReadTrackROOT& adiabatic_tTrackReader = adiabatic_tReader.GetTrack();
    KSReadStepROOT& adiabatic_tStepReader= adiabatic_tReader.GetStep();

    KSReadObjectROOT& adiabatic_tCell = adiabatic_tStepReader.GetObject( "component_step_world" );
    KSDouble& adiabatic_tTime = adiabatic_tCell.Get< KSDouble >( "time" );
    KSThreeVector& adiabatic_tMomentum = adiabatic_tCell.Get< KSThreeVector >( "momentum" );

    vector<Double_t>* adiabatic_time_list = new vector<Double_t>();
    vector<Double_t>* adiabatic_momentum_z_list = new vector<Double_t>();

    for( adiabatic_tRunReader = 0; adiabatic_tRunReader <= adiabatic_tRunReader.GetLastRunIndex(); adiabatic_tRunReader++ )
    {
  		for( adiabatic_tEventReader = adiabatic_tRunReader.GetFirstEventIndex(); adiabatic_tEventReader <= adiabatic_tRunReader.GetLastEventIndex(); adiabatic_tEventReader++ )
  		{
  			for( adiabatic_tTrackReader = adiabatic_tEventReader.GetFirstTrackIndex(); adiabatic_tTrackReader <= adiabatic_tEventReader.GetLastTrackIndex(); adiabatic_tTrackReader++ )
  			{
  				for( adiabatic_tStepReader = adiabatic_tTrackReader.GetFirstStepIndex(); adiabatic_tStepReader <= adiabatic_tTrackReader.GetLastStepIndex(); adiabatic_tStepReader++ )
  				{
  					if( adiabatic_tCell.Valid() )
  					{
              adiabatic_time_list->push_back( adiabatic_tTime.Value() );
              adiabatic_momentum_z_list->push_back( adiabatic_tMomentum.Value().Z() );
  					}
  				}
  			}
  		}
    }

    vector<Double_t>* momentum_diff_list = new vector<Double_t>();

    for ( i = 0; i < 3000; i++ )
    {
        //momentum_diff_list->push_back( 0. );
        momentum_diff_list->push_back( ( (*adiabatic_momentum_z_list)[i] - (*exact_momentum_z_list)[i] ) / (*exact_momentum_z_list)[0] );
        //std::cout << i << "\t" << adiabatic_momentum_z_list->at(i) << "\t" << exact_momentum_z_list->at(i) << "\t" << ( adiabatic_momentum_z_list->at(i) - exact_momentum_z_list->at(i) ) / exact_momentum_z_list->at(i) << "\n";
    }

    //adiabatic_tReader.CloseFile();

    delete adiabatic_tRootFile;

    //delete[] exact_time_list;
    //delete[] exact_momentum_z_list;
    //delete[] adiabatic_time_list;
    //delete[] adiabatic_momentum_z_list;

    TApplication* tApplication = new TApplication("app", 0, 0 );

    TCanvas* c1 = new TCanvas("c1","Track Comparison",200,10,1000,700);

    // std::cout << exact_time_list->size() << "\n";
    // for (i = 0; i < exact_time_list->size(); i++ )
    // {
    //     std::cout << (*exact_time_list)[i] << "\t\t" << (*exact_momentum_z_list)[i] << "\n";
    // }

    //TGraph *gr1 = new TGraph( 3, new double[3] {1.,2.,3.}, new double[3] {3.,2.,1.} );
    // TGraph *gr1 = new TGraph( exact_time_list->size(), &(*exact_time_list)[0], &(*exact_momentum_z_list)[0] );
    // gr1->SetMarkerColor(4);
    // gr1->Draw("AP*");
    //
    // TGraph *gr2 = new TGraph( adiabatic_time_list->size(), &(*adiabatic_time_list)[0], &(*adiabatic_momentum_z_list)[0] );
    // gr2->SetMarkerColor(2);
    // gr2->Draw("P");

    TGraph *gr3 = new TGraph( 3000, &(*adiabatic_time_list)[0], &(*momentum_diff_list)[0] );
    gr3->SetMarkerColor(4);
    gr3->Draw("AP");
    gr3->GetXaxis()->SetTitle( "Time" );
    gr3->GetYaxis()->SetTitle( "Difference in z-Momentum as a Fraction of Initial z-Momentum" );
    gr3->GetXaxis()->CenterTitle();
    gr3->GetYaxis()->CenterTitle();
    gr3->SetTitle( "Momentum Error vs. Time");
    gr3->Draw("AP");

    c1->Update();

    tApplication->Run();

    return 0;
}
