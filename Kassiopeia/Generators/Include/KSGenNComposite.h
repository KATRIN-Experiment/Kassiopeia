#ifndef Kassiopeia_KSGenNComposite_h_
#define Kassiopeia_KSGenNComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenNComposite : public KSComponentTemplate<KSGenNComposite, KSGenCreator>
{
  public:
    KSGenNComposite();
    KSGenNComposite(const KSGenNComposite& aCopy);
    KSGenNComposite* Clone() const override;
    ~KSGenNComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    void SetNValue(KSGenValue* anNValue);
    void ClearNValue(KSGenValue* anNValue);

  private:
    KSGenValue* fNValue;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif
