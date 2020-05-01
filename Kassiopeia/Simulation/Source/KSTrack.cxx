#include "KSTrack.h"

#include "KSComponent.h"

namespace Kassiopeia
{

KSTrack::KSTrack() :
    fTrackId(-1),
    fTrackCount(0),
    fParentEventId(-1),
    fTotalSteps(0),
    fContinuousTime(0.),
    fContinuousLength(0.),
    fContinuousEnergyChange(0.),
    fContinuousMomentumChange(0.),
    fDiscreteSecondaries(0),
    fDiscreteEnergyChange(0.),
    fDiscreteMomentumChange(0.),
    fNumberOfTurns(0),
    fCreatorName(""),
    fTerminatorName(""),
    fInitialParticle(),
    fFinalParticle(),
    fParticleQueue()
{}
KSTrack::KSTrack(const KSTrack& aCopy) :
    KSComponent(),
    fTrackId(aCopy.fTrackId),
    fTrackCount(aCopy.fTrackCount),
    fParentEventId(aCopy.fParentEventId),
    fTotalSteps(aCopy.fTotalSteps),
    fContinuousTime(aCopy.fContinuousTime),
    fContinuousLength(aCopy.fContinuousLength),
    fContinuousEnergyChange(aCopy.fContinuousEnergyChange),
    fContinuousMomentumChange(aCopy.fContinuousMomentumChange),
    fDiscreteSecondaries(aCopy.fDiscreteSecondaries),
    fDiscreteEnergyChange(aCopy.fDiscreteEnergyChange),
    fDiscreteMomentumChange(aCopy.fDiscreteMomentumChange),
    fNumberOfTurns(aCopy.fNumberOfTurns),
    fCreatorName(aCopy.fCreatorName),
    fTerminatorName(aCopy.fTerminatorName),
    fInitialParticle(aCopy.fInitialParticle),
    fFinalParticle(aCopy.fFinalParticle),
    fParticleQueue(aCopy.fParticleQueue)
{}
KSTrack& KSTrack::operator=(const KSTrack& aCopy)
{
    fTrackId = aCopy.fTrackId;
    fTrackCount = aCopy.fTrackCount;
    fParentEventId = aCopy.fParentEventId;
    fTotalSteps = aCopy.fTotalSteps;
    fContinuousTime = aCopy.fContinuousTime;
    fContinuousLength = aCopy.fContinuousLength;
    fContinuousEnergyChange = aCopy.fContinuousEnergyChange;
    fContinuousMomentumChange = aCopy.fContinuousMomentumChange;
    fDiscreteSecondaries = aCopy.fDiscreteSecondaries;
    fDiscreteEnergyChange = aCopy.fDiscreteEnergyChange;
    fDiscreteMomentumChange = aCopy.fDiscreteMomentumChange;
    fNumberOfTurns = aCopy.fNumberOfTurns;
    fCreatorName = aCopy.fCreatorName;
    fTerminatorName = aCopy.fTerminatorName;
    fInitialParticle = aCopy.fInitialParticle;
    fFinalParticle = aCopy.fFinalParticle;
    fParticleQueue = aCopy.fParticleQueue;
    return *this;
}
KSTrack* KSTrack::Clone() const
{
    return new KSTrack(*this);
}
KSTrack::~KSTrack() {}

STATICINT sKSTrackDict =
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetTrackId, "track_id") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetTrackCount, "track_count") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetParentEventId, "parent_event_id") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetTotalSteps, "total_steps") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetContinuousTime, "continuous_time") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetContinuousLength, "continuous_length") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetContinuousEnergyChange, "continuous_energy_change") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetContinuousMomentumChange, "continuous_momentum_change") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetDiscreteSecondaries, "discrete_secondaries") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetDiscreteEnergyChange, "discrete_energy_change") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetDiscreteMomentumChange, "discrete_momentum_change") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetNumberOfTurns, "number_of_turns") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetCreatorName, "creator_name") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetTerminatorName, "terminator_name") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetInitialParticle, "initial_particle") +
    KSDictionary<KSTrack>::AddComponent(&KSTrack::GetFinalParticle, "final_particle");

}  // namespace Kassiopeia
