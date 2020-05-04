#include "KConst.h"
#include "KSReadFileROOT.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

using namespace Kassiopeia;
using namespace std;

int main()
{

    double gyromagnetic_ratio = -1.83247172e+8;
    //double spin_magnitude = 0.5; // in units of hbar

    auto* tRootFile = new KRootFile();
    tRootFile->AddToNames("~/Work/kasper/install/output/Kassiopeia/CombinedNeutronTrapAdiabaticSimulation.root");

    KSReadFileROOT tReader;
    tReader.OpenFile(tRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    KSReadStepROOT& tStepReader = tReader.GetStep();

    KSReadObjectROOT& tCell = tStepReader.GetObject("component_step_world");
    auto& tTime = tCell.Get<KSDouble>("time");
    //KSThreeVector& B = tCell.Get< KSThreeVector >( "magnetic_field" );
    auto& m = tCell.Get<KSDouble>("aligned_spin");

    vector<Double_t> time_list;
    vector<KThreeVector> B_list;

    int cell_count = 0;
    double Bxx_correlation_total = 0., Byy_correlation_total = 0., Bzz_correlation_total = 0.,
           Bxz_correlation_total = 0.;
    double Bxx_correlation, Byy_correlation, Bzz_correlation, Bxz_correlation;

    double theta = 0;
    double frequency_gap = 0.;

    tRunReader = 0;
    tEventReader = tRunReader.GetFirstEventIndex();
    tTrackReader = tEventReader.GetFirstTrackIndex();
    tStepReader = tTrackReader.GetFirstStepIndex();
    if (tCell.Valid()) {
        double initial_m = m.Value();
        theta = acos(initial_m);
        frequency_gap = 1.e10;  //std::fabs( 2. * initial_m * gyromagnetic_ratio * spin_magnitude );
    }
    else {
        std::cout << "FIRST CELL BROKEN";
        exit(0);
    }

    for (; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        for (; tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++) {
            for (; tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++) {
                for (; tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++) {
                    if (tCell.Valid()) {
                        time_list.push_back(tTime.Value());
                        //B_list.push_back( B.Value() );
                        B_list.push_back(KThreeVector(1.,
                                                      std::cos(frequency_gap * tTime.Value()),
                                                      std::sin(frequency_gap * tTime.Value())));
                        //B_list.push_back( KThreeVector( std::sin(1.e+8 * tTime.Value()), std::cos(0.99999e+8 * tTime.Value()), std::sin( 0.9999999e+8 * tTime.Value() ) ) );
                        cell_count++;
                    }
                }
            }
        }
    }

    delete tRootFile;

    std::unordered_map<int, double> exponent_map;

    int average_length = 9999;

    double old_total = 0.;

    for (int i = average_length; i < cell_count - average_length; i++) {
        for (int j = std::max(0, i - average_length); j < std::min(cell_count, i + average_length + 1); j++) {
            double exponent = exponent_map[i - j];
            //std::cout << i << " " << j << " " << exponent << "\n";
            if (exponent == 0) {
                //std::cout << "calculating " << i << " " << j << "\t" << time_list[i] - time_list[j] << "\t" << frequency_gap << " " << frequency_gap * ( time_list[i] - time_list[j] ) << " " << std::cos( frequency_gap * ( time_list[i] - time_list[j] ) ) << "\n";
                exponent = std::cos(frequency_gap * (time_list[i] - time_list[j]));
                exponent_map[i - j] = exponent;
            }
            std::cout << exponent << "\t" << Bxx_correlation_total << "\t";
            Bxx_correlation_total += B_list[i].X() * B_list[j].X() * exponent;
            Byy_correlation_total += B_list[i].Y() * B_list[j].Y() * exponent;
            Bzz_correlation_total += B_list[i].Z() * B_list[j].Z() * exponent;
            Bxz_correlation_total += B_list[i].X() * B_list[j].Z() * exponent;
            std::cout << Bxx_correlation_total << "\n";
        }
        std::cout << Bxx_correlation_total << "\t" << Bxx_correlation_total - old_total << "\n";
        old_total = Bxx_correlation_total;
    }

    Bxx_correlation = Bxx_correlation_total / (2 * average_length + 1) * (time_list[1] - time_list[0]);
    Byy_correlation = Byy_correlation_total / (2 * average_length + 1) * (time_list[1] - time_list[0]);
    Bzz_correlation = Bzz_correlation_total / (2 * average_length + 1) * (time_list[1] - time_list[0]);
    Bxz_correlation = Bxz_correlation_total / (2 * average_length + 1) * (time_list[1] - time_list[0]);

    double TransitionRate =
        1. / 16. * gyromagnetic_ratio * gyromagnetic_ratio *
        (std::cos(theta) * std::cos(theta) * Bxx_correlation + Byy_correlation +
         std::sin(theta) * std::sin(theta) * Bzz_correlation - 2 * std::cos(theta) * std::sin(theta) * Bxz_correlation);

    std::cout << "Resonant frequency: " << frequency_gap << "\n";
    std::cout << "Total correlations: " << Bxx_correlation_total << "  " << Byy_correlation_total << "  "
              << Bzz_correlation_total << "  " << Bxz_correlation_total << "\n";
    std::cout << "    averaging over: " << (2 * average_length + 1) << "\n";
    std::cout << "    at an angle of: " << theta << "\n";
    std::cout << "    with time step: " << time_list[1] - time_list[0] << "\n";
    std::cout << "Transition Rate   : " << TransitionRate << "\n";

    return 0;
}
