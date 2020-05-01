#include "KSReadFileROOT.h"
#include "TApplication.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1.h"

#include <fstream>
#include <iostream>
#include <vector>

using namespace Kassiopeia;

int main()
{

    int term_max_step_count = 0;
    int term_max_r_count = 0;
    int term_minmax_z_count = 0;

    std::vector<double> term_times;

    std::vector<double> initial_ke_values;
    std::vector<double> final_ke_values;

    // string file_base = "~/Work/P8 Tech Note/Center_Standard/";
    // string filelist[] = {"09_09_16__1", "09_09_16__2", "09_10_16__1", "09_11_16__1", "09_12_16__1"};
    // string file_base = "~/Work/P8 Tech Note/Frustrum_Standard/";
    // string filelist[] = {"09_13_16__1", "09_16_16__1", "09_17_16__1"};
    // string file_base = "~/Work/P8 Tech Note/Center_Double_End/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Frustrum_Double_End/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_Half_End/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Frustrum_Half_End/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_1000kA/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_800kA/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_Double_T/";
    // string filelist[] = {"10_03_16", "10_10_16"};
    // string file_base = "~/Work/P8 Tech Note/Center_Half_T/";
    // string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_Half_T/";
    // string filelist[] = {""};
    string file_base = "~/Work/P8 Tech Note/Center_Double_t/";
    string filelist[] = {""};
    // string file_base = "~/Work/P8 Tech Note/Center_Half_T_450kA/";
    // string filelist[] = {""};
    string file_end = "/Project8Simulation.root";

    for (string filename : filelist) {

        auto* tRootFile = new KRootFile();
        tRootFile->AddToNames(file_base + filename + file_end);

        KSReadFileROOT tReader;
        tReader.OpenFile(tRootFile);

        KSReadRunROOT& tRunReader = tReader.GetRun();
        KSReadEventROOT& tEventReader = tReader.GetEvent();
        KSReadTrackROOT& tTrackReader = tReader.GetTrack();
        //KSReadStepROOT& tStepReader= tReader.GetStep();

        //KSReadObjectROOT& tCell = tStepReader.GetObject( "component_track_world" );
        //KSDouble& tTime = tCell.Get< KSDouble >( "time" );
        //KSThreeVector& tMomentum = tCell.Get< KSThreeVector >( "momentum" );

        KSReadObjectROOT& tCell = tTrackReader.GetObject("component_track_world");
        auto& tTerm = tCell.Get<KSString>("terminator_name");

        auto& tTime = tCell.Get<KSDouble>("final_time");
        auto& tIKE = tCell.Get<KSDouble>("initial_kinetic_energy");
        auto& tFKE = tCell.Get<KSDouble>("final_kinetic_energy");

        //vector<Double_t>* list_a = new vector<Double_t>();
        //vector<Double_t>* list_b = new vector<Double_t>();

        for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
            for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
                 tEventReader++) {
                for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                     tTrackReader++) {
                    // for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
                    // {
                    // if( tCell.Valid() && i % 1000 == 999)
                    // {
                    //   //std::cout << "Time: " << tTime.Value() << "\t\tMomentum_Z: " << tMomentum.Value().Z() << "\n";
                    //   time_list->push_back( tTime.Value() );
                    //   momentum_z_list->push_back( tMomentum.Value().Z() );
                    // }
                    // i++;
                    // }

                    if (tCell.Valid()) {
                        if (tTerm.Value() == "term_max_steps") {
                            term_max_step_count++;
                            final_ke_values.push_back(tFKE.Value());
                        }
                        else if (tTerm.Value() == "term_max_r") {
                            term_max_r_count++;
                        }
                        else if (tTerm.Value() == "term_max_z" || tTerm.Value() == "term_min_z") {
                            term_minmax_z_count++;
                        }
                        else {
                            std::cout << "Error: bad terminator name.\n";
                        }
                        term_times.push_back(tTime.Value());
                        initial_ke_values.push_back(tIKE.Value());
                    }
                }
            }
        }

        //tReader.CloseFile();

        delete tRootFile;
    }

    std::cout << "Number trapped: " << term_max_step_count << "\n";

    // TApplication* tApplication = new TApplication("app", 0, 0 );
    //
    // TCanvas* c1 = new TCanvas("c1","Track Comparison",200,10,1000,700);

    //double bar_x[] = {1. ,3. ,5. };
    // char* bar_names[] = {"Trapped", "Escaped Radially", "Escaped Axially"};
    // double bar_values[] = {term_max_step_count, term_max_r_count, term_minmax_z_count};

    // double remaining_count [60];
    // double bar_names [60];
    // for (int i = 0; i < 60; i++){
    //   remaining_count[i] = 100;
    //   bar_names[i] = 0.01 * i;
    // }
    //
    // for (int i = 0; i < term_times.size(); i++){
    //   //std::cout << term_times[i] << "\n";
    //   for (int j = 59; j > 100.0*term_times[i]; j--){
    //     remaining_count[j]--;
    //   }
    // }

    // TGraph *gr1 = new TGraph( 4, bar_names, bar_values );
    // gr1->SetFillColor(40);
    // // gr1->SetMarkerColor(4);
    // gr1->Draw("AB");
    // gr1->GetXaxis()->SetTitle( "Result" );
    // gr1->GetYaxis()->SetTitle( "Frequency" );
    // gr1->GetXaxis()->CenterTitle();
    // gr1->GetYaxis()->CenterTitle();
    // gr1->SetTitle( "Distribution of Simulation Results");
    // gr1->Draw("AB");
    //
    // TH1* h1 = new TH1I("h1", "Distribution of Simulation Results", 3, 0, 6);
    //
    // for( int i = 0; i < 3; i++)
    // {
    //   h1->Fill( bar_names[i], bar_values[i] );
    //   //h1->GetXaxis()->SetBinLabel( i, bar_names[i] );
    // }
    // h1->GetXaxis()->SetTitle("Result");
    // h1->GetYaxis()->SetTitle("Frequency");
    // h1->GetXaxis()->CenterTitle();
    // h1->GetYaxis()->CenterTitle();
    // h1->SetFillColor(40);
    // h1->Draw("HIST");

    // TGraph *gr1 = new TGraph( 4, bar_names, remaining_count );
    // gr1->SetFillColor(40);
    // // gr1->SetMarkerColor(4);
    // gr1->Draw("AB");
    // gr1->GetXaxis()->SetTitle( "Time (s)" );
    // gr1->GetYaxis()->SetTitle( "Particles Still in Trap" );
    // gr1->GetXaxis()->CenterTitle();
    // gr1->GetYaxis()->CenterTitle();
    // gr1->SetTitle( "Particles Trapped as a Function of Time");
    // gr1->Draw("AB");

    // TH1* h1 = new TH1I("h1", "Particles Trapped as a Function of Time", 60, 0, 0.60);
    //
    // for( int i = 0; i < 60; i++)
    // {
    //   h1->Fill( bar_names[i], remaining_count[i] );
    //   //h1->GetXaxis()->SetBinLabel( i, bar_names[i] );
    // }
    // h1->GetXaxis()->SetTitle("Time (s)");
    // h1->GetYaxis()->SetTitle("Particles Still in Trap");
    // h1->GetXaxis()->CenterTitle();
    // h1->GetYaxis()->CenterTitle();
    // h1->SetFillColor(40);
    // h1->Draw("HIST");

    // c1->Update();
    //
    // tApplication->Run();

    std::ofstream TTStream;
    std::ofstream IKEStream;
    std::ofstream FKEStream;

    TTStream.open("Termination_Times.txt");
    for (unsigned int i = 0; i < term_times.size(); i++) {
        TTStream << term_times[i];
        TTStream << "\t";
    }
    TTStream.close();

    IKEStream.open("Initial_Kinetic_Energies.txt");
    for (unsigned int i = 0; i < initial_ke_values.size(); i++) {
        IKEStream << initial_ke_values[i];
        IKEStream << "\t";
    }
    IKEStream.close();

    FKEStream.open("Final_Kinetic_Energies.txt");
    for (unsigned int i = 0; i < final_ke_values.size(); i++) {
        FKEStream << final_ke_values[i];
        FKEStream << "\t";
    }
    FKEStream.close();

    return 0;
}
