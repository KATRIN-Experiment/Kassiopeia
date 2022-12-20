#ifndef Kassiopeia_KSEvent_h_
#define Kassiopeia_KSEvent_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{
class KSRun;

class KSEvent : public KSComponentTemplate<KSEvent>
{
  public:
    KSEvent();
    KSEvent(const KSEvent& aCopy);
    KSEvent& operator=(const KSEvent& aCopy);
    KSEvent* Clone() const override;
    ~KSEvent() override;

    //***
    //IDs
    //***

  public:
    K_REFS(int, EventId)
    K_REFS(int, EventCount)
    K_REFS(int, ParentRunId)

    //*****
    //event
    //*****

  public:
    K_REFS(unsigned int, TotalTracks)
    K_REFS(unsigned int, TotalSteps)
    K_REFS(double, ContinuousTime)
    K_REFS(double, ContinuousLength)
    K_REFS(double, ContinuousEnergyChange)
    K_REFS(double, ContinuousMomentumChange)
    K_REFS(unsigned int, DiscreteSecondaries)
    K_REFS(double, DiscreteEnergyChange)
    K_REFS(double, DiscreteMomentumChange)
    K_REFS(unsigned int, NumberOfTurns)

    //*********
    //generator
    //*********

  public:
    K_REFS(bool, GeneratorFlag)
    K_REFS(std::string, GeneratorName)
    K_REFS(unsigned int, GeneratorPrimaries)
    K_REFS(double, GeneratorEnergy)
    K_REFS(double, GeneratorMinTime)
    K_REFS(double, GeneratorMaxTime)
    K_REFS(katrin::KThreeVector, GeneratorLocation)
    K_REFS(double, GeneratorRadius)

    //*****
    //queue
    //*****

  public:
    K_REFS(KSParticleQueue, ParticleQueue)
};

}  // namespace Kassiopeia

#endif
