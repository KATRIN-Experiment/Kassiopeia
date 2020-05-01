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
    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

    void AddMagneticField(KSMagneticField* aField);

  private:
    void CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField);


  private:
    KSGenValue* fPhiValue;
    KSGenValue* fZValue;
    std::vector<KSMagneticField*> fMagneticFields;
    ;
    K_SET(double, Flux);
    ;
    K_SET(int, NIntegrationSteps);
    ;
    K_SET(bool, OnlySurface);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
