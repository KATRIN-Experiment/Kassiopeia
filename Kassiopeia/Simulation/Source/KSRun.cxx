#include "KSRun.h"

#include "KSDictionary.h"

namespace Kassiopeia
{

    KSRun::KSRun() :
            fRunId( -1 ),
            fRunCount( 0 ),
            fTotalEvents( 0 ),
            fTotalTracks( 0 ),
            fTotalSteps( 0 ),
            fContinuousTime( 0. ),
            fContinuousLength( 0. ),
            fContinuousEnergyChange( 0. ),
            fContinuousMomentumChange( 0. ),
            fDiscreteSecondaries( 0 ),
            fDiscreteEnergyChange( 0. ),
            fDiscreteMomentumChange( 0. ),
            fNumberOfTurns( 0 )
    {
    }
    KSRun::KSRun( const KSRun& aCopy ) :
            KSComponent(),
            fRunId( aCopy.fRunId ),
            fRunCount( aCopy.fRunCount ),
            fTotalEvents( aCopy.fTotalEvents ),
            fTotalTracks( aCopy.fTotalTracks ),
            fTotalSteps( aCopy.fTotalSteps ),
            fContinuousTime( aCopy.fContinuousTime ),
            fContinuousLength( aCopy.fContinuousLength ),
            fContinuousEnergyChange( aCopy.fContinuousEnergyChange ),
            fContinuousMomentumChange( aCopy.fContinuousMomentumChange ),
            fDiscreteSecondaries( aCopy.fDiscreteSecondaries ),
            fDiscreteEnergyChange( aCopy.fDiscreteSecondaries ),
            fDiscreteMomentumChange( aCopy.fDiscreteMomentumChange ),
            fNumberOfTurns( aCopy.fNumberOfTurns )
    {
    }
    KSRun& KSRun::operator=( const KSRun& aCopy )
    {
        fRunId = aCopy.fRunId;
        fRunCount = aCopy.fRunCount;
        fTotalEvents = aCopy.fTotalEvents;
        fTotalTracks = aCopy.fTotalTracks;
        fTotalSteps = aCopy.fTotalSteps;
        fContinuousTime = aCopy.fContinuousTime;
        fContinuousLength = aCopy.fContinuousLength;
        fContinuousEnergyChange = aCopy.fContinuousEnergyChange;
        fContinuousMomentumChange = aCopy.fContinuousMomentumChange;
        fDiscreteSecondaries = aCopy.fDiscreteSecondaries;
        fDiscreteEnergyChange = aCopy.fDiscreteEnergyChange;
        fDiscreteMomentumChange = aCopy.fDiscreteMomentumChange;
        fNumberOfTurns = aCopy.fNumberOfTurns;
        return *this;
    }
    KSRun* KSRun::Clone() const
    {
        return new KSRun( *this );
    }
    KSRun::~KSRun()
    {
    }

    STATICINT sKSRunDict =
        KSDictionary< KSRun >::AddComponent( &KSRun::GetRunId, "run_id" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetRunCount, "run_count" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetTotalEvents, "total_events" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetTotalTracks, "total_tracks" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetTotalSteps, "total_steps" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetContinuousTime, "continuous_time" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetContinuousLength, "continuous_length" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetContinuousEnergyChange, "continuous_energy_change" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetContinuousMomentumChange, "continuous_momentum_change" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetDiscreteSecondaries, "discrete_secondaries" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetDiscreteEnergyChange, "discrete_energy_change" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetDiscreteMomentumChange, "discrete_momentum_change" ) +
        KSDictionary< KSRun >::AddComponent( &KSRun::GetNumberOfTurns, "number_of_turns" );


}
