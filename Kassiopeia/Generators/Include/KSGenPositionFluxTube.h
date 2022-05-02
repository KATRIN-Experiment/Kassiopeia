#ifndef KSC_KSGenPositionFluxTube_h_
#define KSC_KSGenPositionFluxTube_h_

#include "KField.h"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

class KSGenPositionFluxTube : public KSComponentTemplate<KSGenPositionFluxTube, KSGenCreator>
{
  public:
    KSGenPositionFluxTube();
    KSGenPositionFluxTube(const KSGenPositionFluxTube& aCopy);
    KSGenPositionFluxTube* Clone() const override;
    ~KSGenPositionFluxTube() override;

  public:
    void Dice(KSParticleQueue* aPrimaryList) override;

  public:
    void SetOrigin(const katrin::KThreeVector& anOrigin);
    void SetXAxis(const katrin::KThreeVector& anXAxis);
    void SetYAxis(const katrin::KThreeVector& anYAxis);
    void SetZAxis(const katrin::KThreeVector& anZAxis);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

    void AddMagneticField(KSMagneticField* aField);

  private:
    void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                        katrin::KThreeVector& aField);


  private:
    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;

    KSGenValue* fPhiValue;
    KSGenValue* fZValue;
    std::vector<KSMagneticField*> fMagneticFields;

    K_SET(double, Flux);
    K_SET(int, NIntegrationSteps);
    K_SET(bool, OnlySurface);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
