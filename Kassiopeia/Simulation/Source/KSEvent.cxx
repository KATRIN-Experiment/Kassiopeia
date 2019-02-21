#include "KSEvent.h"

namespace Kassiopeia
{

    KSEvent::KSEvent() :
            fEventId( -1 ),
            fEventCount( 0 ),
            fParentRunId( -1 ),
            fTotalTracks( 0 ),
            fTotalSteps( 0 ),
            fContinuousTime( 0. ),
            fContinuousLength( 0. ),
            fContinuousEnergyChange( 0. ),
            fContinuousMomentumChange( 0. ),
            fDiscreteSecondaries( 0 ),
            fDiscreteEnergyChange( 0. ),
            fDiscreteMomentumChange( 0. ),
            fNumberOfTurns( 0 ),
            fGeneratorFlag( false ),
            fGeneratorName( "" ),
            fGeneratorPrimaries( 0 ),
            fGeneratorEnergy( 0. ),
            fGeneratorMinTime( 0. ),
            fGeneratorMaxTime( 0. ),
            fGeneratorLocation( 0., 0., 0. ),
            fGeneratorRadius( 0. ),
            fParticleQueue()
    {
    }
    KSEvent::KSEvent( const KSEvent& aCopy ) :
            KSComponent(),
            fEventId( aCopy.fEventId ),
            fEventCount( aCopy.fEventCount ),
            fParentRunId( aCopy.fParentRunId ),
            fTotalTracks( aCopy.fTotalTracks ),
            fTotalSteps( aCopy.fTotalSteps ),
            fContinuousTime( aCopy.fContinuousTime ),
            fContinuousLength( aCopy.fContinuousLength ),
            fContinuousEnergyChange( aCopy.fContinuousEnergyChange ),
            fContinuousMomentumChange( aCopy.fContinuousMomentumChange ),
            fDiscreteSecondaries( aCopy.fDiscreteSecondaries ),
            fDiscreteEnergyChange( aCopy.fDiscreteSecondaries ),
            fDiscreteMomentumChange( aCopy.fDiscreteMomentumChange ),
            fNumberOfTurns( aCopy.fNumberOfTurns ),
            fGeneratorFlag( aCopy.fGeneratorFlag ),
            fGeneratorName( aCopy.fGeneratorName ),
            fGeneratorPrimaries( aCopy.fGeneratorPrimaries ),
            fGeneratorEnergy( aCopy.fGeneratorEnergy ),
            fGeneratorMinTime( aCopy.fGeneratorMinTime ),
            fGeneratorMaxTime( aCopy.fGeneratorMaxTime ),
            fGeneratorLocation( aCopy.fGeneratorLocation ),
            fGeneratorRadius( aCopy.fGeneratorRadius ),
            fParticleQueue( aCopy.fParticleQueue )
    {
    }
    KSEvent& KSEvent::operator=( const KSEvent& aCopy )
    {
        fEventId = aCopy.fEventId;
        fEventCount = aCopy.fEventCount;
        fParentRunId = aCopy.fParentRunId;
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
        fGeneratorFlag = aCopy.fGeneratorFlag;
        fGeneratorName = aCopy.fGeneratorName;
        fGeneratorPrimaries = aCopy.fGeneratorPrimaries;
        fGeneratorEnergy = aCopy.fGeneratorEnergy;
        fGeneratorMinTime = aCopy.fGeneratorMinTime;
        fGeneratorMaxTime = aCopy.fGeneratorMaxTime;
        fGeneratorLocation = aCopy.fGeneratorLocation;
        fGeneratorRadius = aCopy.fGeneratorRadius;
        fParticleQueue = aCopy.fParticleQueue;
        return *this;
    }
    KSEvent* KSEvent::Clone() const
    {
        return new KSEvent( *this );
    }
    KSEvent::~KSEvent()
    {
    }

    STATICINT sKSEventDict =
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetEventId, "event_id" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetEventCount, "event_count" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetParentRunId, "parent_run_id" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetTotalTracks, "total_tracks" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetTotalSteps, "total_steps" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetContinuousTime, "continuous_time" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetContinuousLength, "continuous_length" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetContinuousEnergyChange, "continuous_energy_change" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetContinuousMomentumChange, "continuous_momentum_change" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetDiscreteSecondaries, "discrete_secondaries" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetDiscreteEnergyChange, "discrete_energy_change" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetDiscreteMomentumChange, "discrete_momentum_change" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetNumberOfTurns, "number_of_turns" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorFlag, "generator_flag" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorName, "generator_name" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorPrimaries, "generator_primaries" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorEnergy, "generator_energy" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorMinTime, "generator_min_time" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorMaxTime, "generator_max_time" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorLocation, "generator_location" ) +
        KSDictionary< KSEvent >::AddComponent( &KSEvent::GetGeneratorRadius, "generator_radius" );

}
