#include "KSReadFileROOT.h"

#include <limits>
using std::numeric_limits;

using namespace Kassiopeia;

int main()
{
    katrin::KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    katrin::KMessageTable::GetInstance()->SetLogVerbosity( eDebug );

    KRootFile* tRootFile = CreateOutputRootFile( "QuadrupoleTrapSimulation.root" );

    KSReadFileROOT tReader;
    tReader.OpenFile( tRootFile );

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    KSReadStepROOT& tStepReader= tReader.GetStep();

    KSReadObjectROOT& tWorld = tStepReader.GetObject( "component_step_world" );
//    KSDouble& tLength = tWorld.Get< KSDouble >( "time" );

    KSReadObjectROOT& tCell = tStepReader.GetObject( "component_step_cell" );
    KSDouble& tMoment = tCell.Get< KSDouble >( "orbital_magnetic_moment" );
    KSDouble tMinMoment;
    KSDouble tMaxMoment;
    double tDeviation;

    for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
    {
		for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
		{
			for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
			{
				tMinMoment = numeric_limits< double >::max();
				tMaxMoment = numeric_limits< double >::min();
				for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
				{
					if( tCell.Valid() )
					{
						if( tMoment.Value() > tMaxMoment.Value() )
						{
							tMaxMoment = tMoment;
						}

						if( tMoment.Value() < tMinMoment.Value() )
						{
							tMinMoment = tMoment;
						}
					}
				}

				tDeviation = 2.0 * ((tMaxMoment.Value() - tMinMoment.Value()) / (tMaxMoment.Value() + tMinMoment.Value()));

				cout << "extrema for track <" << tDeviation << ">" << endl;
			}
		}
    }


    tReader.CloseFile();

    delete tRootFile;

    return 0;
}
