#ifndef Kassiopeia_KSModSplitOnTurn_h_
#define Kassiopeia_KSModSplitOnTurn_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSStepModifier.h"

namespace Kassiopeia
{

class KSModSplitOnTurn : public KSComponentTemplate<KSModSplitOnTurn, KSStepModifier>
{
  public:
    enum
    {
        // use binary numbers here (allows combinations like `eForward | eBackward`)
        eForward = 0b0001,
        eBackward = 0b0010,
    };

  public:
    KSModSplitOnTurn();
    KSModSplitOnTurn(const KSModSplitOnTurn& aCopy);
    KSModSplitOnTurn* Clone() const override;
    ~KSModSplitOnTurn() override;

  public:
    bool ExecutePreStepModification(KSParticle& anInitialParticle, KSParticleQueue& aQueue) override;
    bool ExecutePostStepModification(KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                     KSParticleQueue& aQueue) override;

  public:
    K_SET_GET(int, Direction);

  private:
    double fCurrentDotProduct;

  private:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  protected:
    void PullDeupdateComponent() override;
    void PushDeupdateComponent() override;
};
}  // namespace Kassiopeia

#endif
