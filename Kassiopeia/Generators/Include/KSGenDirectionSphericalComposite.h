#ifndef Kassiopeia_KSGenDirectionSphericalComposite_h_
#define Kassiopeia_KSGenDirectionSphericalComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenDirectionSphericalComposite : public KSComponentTemplate<KSGenDirectionSphericalComposite, KSGenCreator>
{
  public:
    KSGenDirectionSphericalComposite();
    KSGenDirectionSphericalComposite(const KSGenDirectionSphericalComposite& aCopy);
    KSGenDirectionSphericalComposite* Clone() const override;
    ~KSGenDirectionSphericalComposite() override;

  public:
    void Dice(KSParticleQueue* aParticleList) override;

  public:
    void SetXAxis(const KGeoBag::KThreeVector& anXAxis);
    void SetYAxis(const KGeoBag::KThreeVector& anYAxis);
    void SetZAxis(const KGeoBag::KThreeVector& anZAxis);

    void SetThetaValue(KSGenValue* anThetaValue);
    void ClearThetaValue(KSGenValue* anThetaValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

  private:
    KSGenValue* fThetaValue;
    KSGenValue* fPhiValue;

    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
