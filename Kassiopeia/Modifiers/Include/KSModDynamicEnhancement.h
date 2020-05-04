#ifndef Kassiopeia_KSModDynamicEnhancement_h_
#define Kassiopeia_KSModDynamicEnhancement_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSIntScattering.h"
#include "KSStepModifier.h"
#include "KSTrajTermSynchrotron.h"

namespace Kassiopeia
{

class KSModDynamicEnhancement : public KSComponentTemplate<KSModDynamicEnhancement, KSStepModifier>
{
  public:
    KSModDynamicEnhancement();
    KSModDynamicEnhancement(const KSModDynamicEnhancement& aCopy);
    KSModDynamicEnhancement* Clone() const override;
    ~KSModDynamicEnhancement() override;

  public:
    bool ExecutePreStepModification(KSParticle& anInitialParticle, KSParticleQueue& aQueue) override;
    bool ExecutePostStepModification(KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                     KSParticleQueue& aQueue) override;

  public:
    K_GET(double, Enhancement)
    K_SET_GET(double, StaticEnhancement)
    K_SET_GET(bool, Dynamic)
    K_SET_GET(double, ReferenceCrossSectionEnergy)

  public:
    void SetScattering(KSIntScattering* aScattering);
    void SetSynchrotron(KSTrajTermSynchrotron* aSynchrotron);

  private:
    KSIntScattering* fScattering;
    KSTrajTermSynchrotron* fSynchrotron;
    double fReferenceCrossSection;

  private:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  protected:
    void PullDeupdateComponent() override;
    void PushDeupdateComponent() override;
};
}  // namespace Kassiopeia

#endif
