#ifndef Kassiopeia_KSRootSurfaceNavigator_h_
#define Kassiopeia_KSRootSurfaceNavigator_h_

#include "KMathBracketingSolver.h"
#include "KSList.h"
#include "KSStep.h"
#include "KSSurfaceNavigator.h"
#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSRootSurfaceNavigator : public KSComponentTemplate<KSRootSurfaceNavigator, KSSurfaceNavigator>
{
  public:
    KSRootSurfaceNavigator();
    KSRootSurfaceNavigator(const KSRootSurfaceNavigator& aCopy);
    KSRootSurfaceNavigator* Clone() const override;
    ~KSRootSurfaceNavigator() override;

    //******************
    //surface navigation
    //******************

  protected:
    void ExecuteNavigation(const KSParticle& anInitialParticle, const KSParticle& anNavigationParticle,
                           KSParticle& aFinalParticle, KSParticleQueue& aSecondaries) const override;
    void FinalizeNavigation(KSParticle& aFinalParticle) const override;

    //***********
    //composition
    //***********

  public:
    void SetSurfaceNavigator(KSSurfaceNavigator* anNavigation);
    void ClearSurfaceNavigator(KSSurfaceNavigator* anNavigation);

  private:
    KSSurfaceNavigator* fSurfaceNavigator;

    //******
    //action
    //******

  public:
    void SetStep(KSStep* anStep);

    void ExecuteNavigation();
    void FinalizeNavigation();

  private:
    KSStep* fStep;
    const KSParticle* fTerminatorParticle;
    const KSParticle* fInteractionParticle;
    KSParticle* fFinalParticle;
    KSParticleQueue* fParticleQueue;
};

}  // namespace Kassiopeia

#endif
