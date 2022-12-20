#include "KSReadFileROOT.h"

#include <limits>
using std::numeric_limits;

using namespace Kassiopeia;

int main()
{
    using std::cout;
    using std::endl;

    katrin::KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    katrin::KMessageTable::GetInstance().SetLogVerbosity(eDebug);

    auto tRootFile = katrin::KRootFile::CreateOutputRootFile("QuadrupoleTrapSimulation.root");

    KSReadFileROOT tReader;
    tReader.OpenFile(tRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    KSReadStepROOT& tStepReader = tReader.GetStep();

    //    KSReadObjectROOT& tWorld = tStepReader.Get( "component_step_world" );
    //    KSDouble& tLength = tWorld.Get< KSDouble >( "time" );

    KSReadObjectROOT& tCell = tStepReader.GetObject("component_step_cell");
    auto& tPosition = tCell.Get<KSThreeVector>("guiding_center_position");
    auto& tMoment = tCell.Get<KSDouble>("orbital_magnetic_moment");
    KSDouble tMinMoment;
    KSDouble tMaxMoment;
    double tDeviation;

    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                tMinMoment = numeric_limits<double>::max();
                tMaxMoment = numeric_limits<double>::min();
                cout << tTrackReader.GetLastStepIndex()-tTrackReader.GetFirstStepIndex() << endl;
                size_t tSteps = 0;
                for (tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex();
                     tStepReader++) {
                    if (tCell.Valid()) {
                        if (tSteps == 0) {
                            cout << "first valid: " << tStepReader.GetStepIndex() << endl;
                            cout << "first position: " << tPosition.Value() << endl;
                            cout << "first value: " << tMoment.Value() << endl;
                        }
                        tSteps++;
                        if (tMoment.Value() > tMaxMoment.Value()) {
                            tMaxMoment = tMoment;
                        }

                        if (tMoment.Value() < tMinMoment.Value()) {
                            tMinMoment = tMoment;
                        }
                    }
                }

                cout << "last valid: " << tStepReader.GetStepIndex() << endl;
                cout << "last position: " << tPosition.Value() << endl;
                cout << "last value: " << tMoment.Value() << endl;

                cout << tSteps << " steps" << endl;
                cout << "from " << tTrackReader.GetFirstStepIndex() << " to " << tTrackReader.GetLastStepIndex() << endl;

                cout << "max: " << tMaxMoment.Value() << ", min: " << tMinMoment.Value() << endl;

                tDeviation = 2.0 * ((tMaxMoment.Value() - tMinMoment.Value()) / (tMaxMoment.Value() + tMinMoment.Value()));

                cout << "extrema for track #" << tTrackReader.GetTrackIndex() << ": <" << tDeviation << ">" << endl;
                cout << endl;
                //break;
            }
        }
    }


    tReader.CloseFile();

    delete tRootFile;

    return 0;
}
