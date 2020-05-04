#include "KSMainMessage.h"
#include "KSReadFileROOT.h"
#include "TApplication.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TMultiGraph.h"

#include <cmath>
#include <fstream>
#include <getopt.h>
#include <stdlib.h>


using namespace Kassiopeia;


int main(int argc, char** argv)
{
    katrin::KMessageTable::GetInstance()->SetTerminalVerbosity(eDebug);
    katrin::KMessageTable::GetInstance()->SetLogVerbosity(eNormal);

    TApplication* tApplication = new TApplication("PlotTransmission", 0, 0);

    if (argc < 2) {
        mainmsg(eError) << "Usage: SimulationTransmissionFunction <inputfile>  " << eom;
    }

    std::string tFileName = std::string(argv[1]);

    double edge = 782.76;
    double upper = edge + 3.0;
    double lower = edge - 3.0;

    double tNumBinsTransmission = 30;
    double tMinInitialEnergy = 0.0;
    double tMaxInitialEnergy = 0.0;

    KRootFile* tRootFile = new KRootFile();
    tRootFile->AddToNames(tFileName);

    KSReadFileROOT tReader;
    tReader.OpenFile(tRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();

    //groups
    KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject("component_track_world");
    KSDouble& tInitialEnergy = tTrackGroup.Get<KSDouble>("initial_kinetic_energy");
    KSDouble& tFinalEnergy = tTrackGroup.Get<KSDouble>("final_kinetic_energy");
    KSString& tCreatorName = tTrackGroup.Get<KSString>("creator_name");
    KSString& tTerminatorName = tTrackGroup.Get<KSString>("terminator_name");
    KSThreeVector& tInitialPosition = tTrackGroup.Get<KSThreeVector>("initial_position");
    KSThreeVector& tFinalPosition = tTrackGroup.Get<KSThreeVector>("final_position");
    KSThreeVector& tInitialMagneticField = tTrackGroup.Get<KSThreeVector>("initial_magnetic_field");
    KSThreeVector& tFinalMagneticField = tTrackGroup.Get<KSThreeVector>("final_magnetic_field");
    KSDouble& tInitialTime = tTrackGroup.Get<KSDouble>("initial_time");
    KSDouble& tFinalTime = tTrackGroup.Get<KSDouble>("final_time");
    KSDouble& tInitialPotential = tTrackGroup.Get<KSDouble>("initial_electric_potential");
    KSDouble& tFinalPotential = tTrackGroup.Get<KSDouble>("final_electric_potential");
    KSDouble& tInitialPolarAngle = tTrackGroup.Get<KSDouble>("initial_polar_angle_to_b");
    KSDouble& tFinalPolarAngle = tTrackGroup.Get<KSDouble>("final_polar_angle_to_b");

    //find maximum and minimum initial particle energies for calculating the bins
    //initialise max and min energies with first energy value
    mainmsg(eNormal) << "Getting minimal and maximal initial energy..." << eom;

    tEventReader = tRunReader.GetFirstEventIndex();
    tTrackReader = tEventReader.GetFirstTrackIndex();
    tMaxInitialEnergy = tInitialEnergy.Value();
    tMinInitialEnergy = tMaxInitialEnergy;
    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        //mainmsg( eDebug ) << "Analyzing Run #" << tRunReader.Index()+1 << " of " << tRunReader.GetLastRunIndex()+1 << eom;
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                if (tTrackGroup.Valid() && tCreatorName.Value() == "fake_egun") {
                    if (tInitialEnergy.Value() > tMaxInitialEnergy) {
                        tMaxInitialEnergy = tInitialEnergy.Value();
                    }
                    else if (tInitialEnergy.Value() < tMinInitialEnergy) {
                        tMinInitialEnergy = tInitialEnergy.Value();
                    }
                }
            }
        }
    }

    // double tBinWidthTransmission = (tMaxInitialEnergy - tMinInitialEnergy) / tNumBinsTransmission;
    double tBinWidthTransmission = (upper - lower) / tNumBinsTransmission;
    TH1D* transmission_histo = new TH1D("transmitted",
                                        "transmitted",
                                        tNumBinsTransmission + 2,
                                        lower - tBinWidthTransmission,
                                        upper + tBinWidthTransmission);
    TH1D* produced_histo = new TH1D("produced",
                                    "produced",
                                    tNumBinsTransmission + 2,
                                    lower - tBinWidthTransmission,
                                    upper + tBinWidthTransmission);
    TH1D* tof_histo = new TH1D("tof", "tof", 50, 0, 10e-6);


    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        //mainmsg( eDebug ) << "Calculating run #" << tRunReader.Index()+1 << " of " << tRunReader.GetLastRunIndex()+1 << eom;
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                if (tTrackGroup.Valid()) {
                    //increment production histogram
                    produced_histo->Fill(tInitialEnergy.Value());

                    // //now determine, if the particle was transmitted
                    // if( tTerminatorName.Value() == "term_max_z"  )
                    // {
                    //     transmission_histo->Fill(tInitialEnergy.Value());
                    // }

                    //now determine, if the particle was transmitted
                    if ((tFinalPosition.Value()).Z() > 12.0) {
                        transmission_histo->Fill(tInitialEnergy.Value());
                    }


                    if ((tFinalPosition.Value()).Z() > 12.0) {
                        if (tInitialEnergy.Value() < 780.8 && tInitialEnergy.Value() < 781) {
                            double tof = tFinalTime.Value();
                            int n_period = std::floor((tof / 10e-6));
                            double mod_tof = std::fabs(tof - n_period * 10e-6);
                            tof_histo->Fill(mod_tof);
                        }
                    }
                }
            }
        }
    }

    TGraph* tf_plot = new TGraph();

    TCanvas* tCanvasTransmission = new TCanvas("tCanvasTransmission", "transmission probability");
    tCanvasTransmission->Connect("Closed()", "TApplication", tApplication, "Terminate()");
    tCanvasTransmission->cd();
    tCanvasTransmission->SetBottomMargin(0.12);
    tCanvasTransmission->SetLeftMargin(0.12);
    tCanvasTransmission->SetGrid();
    tCanvasTransmission->Divide(2, 2);

    tCanvasTransmission->cd(1);


    for (unsigned int i = 0; i < transmission_histo->GetNbinsX(); i++) {
        double tf_val = transmission_histo->GetBinContent(i) / (produced_histo->GetBinContent(i) + 1);
        double energy_val = transmission_histo->GetBinCenter(i);

        tf_plot->SetPoint(i, energy_val, tf_val);
    }


    tf_plot->SetMarkerStyle(20);
    tf_plot->SetMarkerSize(0.5);
    tf_plot->GetXaxis()->SetTitle("E in eV");
    tf_plot->GetXaxis()->SetLabelSize(0.05);
    tf_plot->GetXaxis()->SetTitleSize(0.05);
    tf_plot->GetYaxis()->SetTitle("transmission probability");
    tf_plot->GetYaxis()->SetLabelSize(0.05);
    tf_plot->GetYaxis()->SetTitleSize(0.05);


    tf_plot->Draw("ALP");

    tCanvasTransmission->cd(2);

    transmission_histo->Draw();

    tCanvasTransmission->cd(3);

    tof_histo->Draw();

    tApplication->Run();

    return 0;
}
