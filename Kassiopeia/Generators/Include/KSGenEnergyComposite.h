#ifndef Kassiopeia_KSGenEnergyComposite_h_
#define Kassiopeia_KSGenEnergyComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenEnergyComposite : public KSComponentTemplate<KSGenEnergyComposite, KSGenCreator>
{
  public:
    KSGenEnergyComposite();
    KSGenEnergyComposite(const KSGenEnergyComposite& aCopy);
    KSGenEnergyComposite* Clone() const override;
    ~KSGenEnergyComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    void SetEnergyValue(KSGenValue* anEnergyValue);
    void ClearEnergyValue(KSGenValue* anEnergyValue);

  private:
    KSGenValue* fEnergyValue;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif
