#ifndef Kassiopeia_KSGenSpinRelativeComposite_h_
#define Kassiopeia_KSGenSpinRelativeComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenSpinRelativeComposite : public KSComponentTemplate<KSGenSpinRelativeComposite, KSGenCreator>
{
  public:
    KSGenSpinRelativeComposite();
    KSGenSpinRelativeComposite(const KSGenSpinRelativeComposite& aCopy);
    KSGenSpinRelativeComposite* Clone() const override;
    ~KSGenSpinRelativeComposite() override;

  public:
    void Dice(KSParticleQueue* aParticleList) override;

  public:
    void SetThetaValue(KSGenValue* anThetaValue);
    void ClearThetaValue(KSGenValue* anThetaValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

  private:
    KSGenValue* fThetaValue;
    KSGenValue* fPhiValue;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
