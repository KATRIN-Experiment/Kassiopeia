#ifndef Kassiopeia_KSTrack_h_
#define Kassiopeia_KSTrack_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{
class KSEvent;

class KSTrack : public KSComponentTemplate<KSTrack>
{
  public:
    KSTrack();
    KSTrack(const KSTrack& aCopy);
    KSTrack& operator=(const KSTrack& aCopy);
    KSTrack* Clone() const override;
    ~KSTrack() override;

    //***
    //IDs
    //***

  public:
    K_REFS(int, TrackId)
    K_REFS(int, TrackCount)
    K_REFS(int, ParentEventId)

    //*****
    //track
    //*****

  public:
    K_REFS(unsigned int, TotalSteps)
    K_REFS(double, ContinuousTime)
    K_REFS(double, ContinuousLength)
    K_REFS(double, ContinuousEnergyChange)
    K_REFS(double, ContinuousMomentumChange)
    K_REFS(unsigned int, DiscreteSecondaries)
    K_REFS(double, DiscreteEnergyChange)
    K_REFS(double, DiscreteMomentumChange)
    K_REFS(unsigned int, NumberOfTurns)
    K_REFS(std::string, CreatorName)
    K_REFS(std::string, TerminatorName)

    //*********
    //particles
    //*********

  public:
    K_REFS(KSParticle, InitialParticle)
    K_REFS(KSParticle, FinalParticle)

    //*****
    //queue
    //*****

  public:
    K_REFS(KSParticleQueue, ParticleQueue)
};

}  // namespace Kassiopeia

#endif
