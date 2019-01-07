#include "KSStep.h"

#include "KSComponent.h"

namespace Kassiopeia
{

    KSStep::KSStep() :
            fStepId( -1 ),
            fStepCount( 0 ),
            fParentTrackId( -1 ),
            fContinuousTime( 0. ),
            fContinuousLength( 0. ),
            fContinuousEnergyChange( 0. ),
            fContinuousMomentumChange( 0. ),
            fDiscreteSecondaries( 0 ),
            fDiscreteEnergyChange( 0. ),
            fDiscreteMomentumChange( 0. ),
            fNumberOfTurns( 0 ),
            fModifierName( "" ),
            fModifierFlag( false ),
            fTerminatorName( "" ),
            fTerminatorFlag( false ),
            fTrajectoryName( "" ),
            fTrajectoryCenter( 0., 0., 0. ),
            fTrajectoryRadius( 0. ),
            fTrajectoryStep( 0. ),
            fSpaceInteractionName( "" ),
            fSpaceInteractionStep( 0. ),
            fSpaceInteractionFlag( false ),
            fSpaceNavigationName( "" ),
            fSpaceNavigationStep( 0. ),
            fSpaceNavigationFlag( false ),
            fSurfaceInteractionName( "" ),
            fSurfaceInteractionFlag( false ),
            fSurfaceNavigationName( "" ),
            fSurfaceNavigationFlag( false ),
            fInitialParticle(),
            fTerminatorParticle(),
            fTrajectoryParticle(),
            fInteractionParticle(),
            fNavigationParticle(),
            fFinalParticle(),
            fParticleQueue()
    {
    }
    KSStep::KSStep( const KSStep& aCopy ) :
            KSComponent(),
            fStepId( aCopy.fStepId ),
            fStepCount( aCopy.fStepCount ),
            fParentTrackId( aCopy.fParentTrackId ),
            fContinuousTime( aCopy.fContinuousTime ),
            fContinuousLength( aCopy.fContinuousLength ),
            fContinuousEnergyChange( aCopy.fContinuousEnergyChange ),
            fContinuousMomentumChange( aCopy.fContinuousMomentumChange ),
            fDiscreteSecondaries( aCopy.fDiscreteSecondaries ),
            fDiscreteEnergyChange( aCopy.fDiscreteSecondaries ),
            fDiscreteMomentumChange( aCopy.fDiscreteMomentumChange ),
            fNumberOfTurns( aCopy.fNumberOfTurns ),
            fModifierName( aCopy.fModifierName ),
            fModifierFlag( aCopy.fModifierFlag ),
            fTerminatorName( aCopy.fTerminatorName ),
            fTerminatorFlag( aCopy.fTerminatorFlag ),
            fTrajectoryName( aCopy.fTrajectoryName ),
            fTrajectoryCenter( aCopy.fTrajectoryCenter ),
            fTrajectoryRadius( aCopy.fTrajectoryRadius ),
            fTrajectoryStep( aCopy.fTrajectoryStep ),
            fSpaceInteractionName( aCopy.fSpaceInteractionName ),
            fSpaceInteractionStep( aCopy.fSpaceInteractionStep ),
            fSpaceInteractionFlag( aCopy.fSpaceInteractionFlag ),
            fSpaceNavigationName( aCopy.fSpaceNavigationName ),
            fSpaceNavigationStep( aCopy.fSpaceNavigationStep ),
            fSpaceNavigationFlag( aCopy.fSpaceNavigationFlag ),
            fSurfaceInteractionName( aCopy.fSurfaceInteractionName ),
            fSurfaceInteractionFlag( aCopy.fSurfaceInteractionFlag ),
            fSurfaceNavigationName( aCopy.fSurfaceNavigationName ),
            fSurfaceNavigationFlag( aCopy.fSurfaceNavigationFlag ),
            fInitialParticle( aCopy.fInitialParticle ),
            fTerminatorParticle( aCopy.fTerminatorParticle ),
            fTrajectoryParticle( aCopy.fTrajectoryParticle ),
            fInteractionParticle( aCopy.fInteractionParticle ),
            fNavigationParticle( aCopy.fNavigationParticle ),
            fFinalParticle( aCopy.fFinalParticle ),
            fParticleQueue( aCopy.fParticleQueue )
    {
    }
    KSStep& KSStep::operator=( const KSStep& aCopy )
    {
        fStepId = aCopy.fStepId;
        fStepCount = aCopy.fStepCount;
        fParentTrackId = aCopy.fParentTrackId;
        fContinuousTime = aCopy.fContinuousTime;
        fContinuousLength = aCopy.fContinuousLength;
        fContinuousEnergyChange = aCopy.fContinuousEnergyChange;
        fContinuousMomentumChange = aCopy.fContinuousMomentumChange;
        fDiscreteSecondaries = aCopy.fDiscreteSecondaries;
        fDiscreteEnergyChange = aCopy.fDiscreteEnergyChange;
        fDiscreteMomentumChange = aCopy.fDiscreteMomentumChange;
        fNumberOfTurns = aCopy.fNumberOfTurns;
        fModifierName = aCopy.fModifierName;
        fModifierFlag = aCopy.fModifierFlag;
        fTerminatorName = aCopy.fTerminatorName;
        fTerminatorFlag = aCopy.fTerminatorFlag;
        fTrajectoryName = aCopy.fTrajectoryName;
        fTrajectoryCenter = aCopy.fTrajectoryCenter;
        fTrajectoryRadius = aCopy.fTrajectoryRadius;
        fTrajectoryStep = aCopy.fTrajectoryStep;
        fSpaceInteractionName = aCopy.fSpaceInteractionName;
        fSpaceInteractionStep = aCopy.fSpaceInteractionStep;
        fSpaceInteractionFlag = aCopy.fSpaceInteractionFlag;
        fSpaceNavigationName = aCopy.fSpaceNavigationName;
        fSpaceNavigationStep = aCopy.fSpaceNavigationStep;
        fSpaceNavigationFlag = aCopy.fSpaceNavigationFlag;
        fSurfaceInteractionName = aCopy.fSurfaceInteractionName;
        fSurfaceInteractionFlag = aCopy.fSurfaceInteractionFlag;
        fSurfaceNavigationName = aCopy.fSurfaceNavigationName;
        fSurfaceNavigationFlag = aCopy.fSurfaceNavigationFlag;
        fInitialParticle = aCopy.fInitialParticle;
        fTerminatorParticle = aCopy.fTerminatorParticle;
        fTrajectoryParticle = aCopy.fTrajectoryParticle;
        fNavigationParticle = aCopy.fNavigationParticle;
        fInteractionParticle = aCopy.fInteractionParticle;
        fFinalParticle = aCopy.fFinalParticle;
        fParticleQueue = aCopy.fParticleQueue;
        return *this;
    }
    KSStep* KSStep::Clone() const
    {
        return new KSStep( *this );
    }
    KSStep::~KSStep()
    {
    }

    STATICINT sKSStepDict =
        KSDictionary< KSStep >::AddComponent( &KSStep::GetStepId, "step_id" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetStepCount, "step_count" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetParentTrackId, "parent_track_id" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetContinuousTime, "continuous_time" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetContinuousLength, "continuous_length" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetContinuousEnergyChange, "continuous_energy_change" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetContinuousMomentumChange, "continuous_momentum_change" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetDiscreteSecondaries, "discrete_secondaries" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetDiscreteEnergyChange, "discrete_energy_change" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetDiscreteMomentumChange, "discrete_momentum_change" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetNumberOfTurns, "number_of_turns" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetModifierName, "modifier_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetModifierFlag, "modifier_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTerminatorName, "terminator_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTerminatorFlag, "terminator_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTrajectoryName, "trajectory_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTrajectoryCenter, "trajectory_center" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTrajectoryRadius, "trajectory_radius" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTrajectoryStep, "trajectory_step" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceInteractionName, "space_interaction_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceInteractionStep, "space_interaction_step" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceInteractionFlag, "space_interaction_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceNavigationName, "space_navigation_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceNavigationStep, "space_navigation_step" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSpaceNavigationFlag, "space_navigation_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSurfaceInteractionName, "surface_interaction_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSurfaceInteractionFlag, "surface_interaction_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSurfaceNavigationName, "surface_navigation_name" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetSurfaceNavigationFlag, "surface_navigation_flag" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetInitialParticle, "initial_particle" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTerminatorParticle, "terminator_particle" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetTrajectoryParticle, "trajectory_particle" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetInteractionParticle, "interaction_particle" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetNavigationParticle, "navigation_particle" ) +
        KSDictionary< KSStep >::AddComponent( &KSStep::GetFinalParticle, "final_particle" );
}
