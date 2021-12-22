#ifndef Kassiopeia_KSRootSpaceInteraction_h_
#define Kassiopeia_KSRootSpaceInteraction_h_

#include "KMathBracketingSolver.h"
#include "KSList.h"
#include "KSSpaceInteraction.h"
#include "KSStep.h"
#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSRootSpaceInteraction : public KSComponentTemplate<KSRootSpaceInteraction, KSSpaceInteraction>
{
  public:
    KSRootSpaceInteraction();
    KSRootSpaceInteraction(const KSRootSpaceInteraction& aCopy);
    KSRootSpaceInteraction* Clone() const override;
    ~KSRootSpaceInteraction() override;

    //*****************
    //space interaction
    //*****************

  protected:
    void CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                              const KSParticle& aTrajectoryFinalParticle,
                              const katrin::KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius,
                              const double& aTrajectoryTimeStep, KSParticle& anInteractionParticle, double& aTimeStep,
                              bool& aFlag) override;
    void ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) const override;

    //***********
    //composition
    //***********

  public:
    void AddSpaceInteraction(KSSpaceInteraction* anInteraction);
    void RemoveSpaceInteraction(KSSpaceInteraction* anInteraction);

  private:
    KSList<KSSpaceInteraction> fSpaceInteractions;
    KSSpaceInteraction* fSpaceInteraction;

    //******
    //action
    //******

  public:
    void SetStep(KSStep* anStep);
    void SetTrajectory(KSTrajectory* aTrajectory);

    void CalculateInteraction();
    void ExecuteInteraction();

    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;

  private:
    KSStep* fStep;
    const KSParticle* fTerminatorParticle;
    const KSParticle* fTrajectoryParticle;
    KSParticle* fInteractionParticle;
    KSParticle* fFinalParticle;
    KSParticleQueue* fParticleQueue;
    KSTrajectory* fTrajectory;
};

}  // namespace Kassiopeia

#endif
