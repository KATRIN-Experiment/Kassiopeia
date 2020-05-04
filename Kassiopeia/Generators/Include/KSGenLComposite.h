#ifndef Kassiopeia_KSGenLComposite_h_
#define Kassiopeia_KSGenLComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenLComposite : public KSComponentTemplate<KSGenLComposite, KSGenCreator>
{
  public:
    KSGenLComposite();
    KSGenLComposite(const KSGenLComposite& aCopy);
    KSGenLComposite* Clone() const override;
    ~KSGenLComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    void SetLValue(KSGenValue* anLValue);
    void ClearLValue(KSGenValue* anLValue);

  private:
    KSGenValue* fLValue;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif
