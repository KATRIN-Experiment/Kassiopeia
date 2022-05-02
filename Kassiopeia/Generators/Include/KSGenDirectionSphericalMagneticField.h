#ifndef Kassiopeia_KSGenDirectionSphericalMagneticField_h_
#define Kassiopeia_KSGenDirectionSphericalMagneticField_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenDirectionSphericalMagneticField : public KSComponentTemplate<KSGenDirectionSphericalMagneticField, KSGenCreator>
{
  public:
    KSGenDirectionSphericalMagneticField();
    KSGenDirectionSphericalMagneticField(const KSGenDirectionSphericalMagneticField& aCopy);
    KSGenDirectionSphericalMagneticField* Clone() const override;
    ~KSGenDirectionSphericalMagneticField() override;

  public:
    void Dice(KSParticleQueue* aParticleList) override;

  public:
    void AddMagneticField(KSMagneticField* aField);

    void SetThetaValue(KSGenValue* anThetaValue);
    void ClearThetaValue(KSGenValue* anThetaValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

  private:
    void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime, katrin::KThreeVector& aField);
    static void BuildCoordinateSystem(katrin::KThreeVector& u, katrin::KThreeVector& v, katrin::KThreeVector& w, const katrin::KThreeVector& n);

  private:
    std::vector<KSMagneticField*> fMagneticFields;

    KSGenValue* fThetaValue;
    KSGenValue* fPhiValue;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
