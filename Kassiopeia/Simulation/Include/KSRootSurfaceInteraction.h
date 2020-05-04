#ifndef _Kassiopeia_KSRootSurfaceInteraction_h_
#define _Kassiopeia_KSRootSurfaceInteraction_h_

#include "KSList.h"
#include "KSStep.h"
#include "KSSurfaceInteraction.h"

namespace Kassiopeia
{

class KSRootSurfaceInteraction : public KSComponentTemplate<KSRootSurfaceInteraction, KSSurfaceInteraction>
{
  public:
    KSRootSurfaceInteraction();
    KSRootSurfaceInteraction(const KSRootSurfaceInteraction& aCopy);
    KSRootSurfaceInteraction* Clone() const override;
    ~KSRootSurfaceInteraction() override;

    //*******************
    //surface interaction
    //*******************

  protected:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

    //***********
    //composition
    //***********

  public:
    void SetSurfaceInteraction(KSSurfaceInteraction* anInteraction);
    void ClearSurfaceInteraction(KSSurfaceInteraction* anInteraction);

  private:
    KSSurfaceInteraction* fSurfaceInteraction;

    //******
    //action
    //******

  public:
    void SetStep(KSStep* anStep);

    void ExecuteInteraction();

  private:
    KSStep* fStep;
    const KSParticle* fTerminatorParticle;
    KSParticle* fInteractionParticle;
    KSParticleQueue* fParticleQueue;
};

}  // namespace Kassiopeia

#endif
