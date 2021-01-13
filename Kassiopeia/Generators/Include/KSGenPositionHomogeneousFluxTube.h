#ifndef KSC_KSGenPositionHomogeneousFluxTube_h_
#define KSC_KSGenPositionHomogeneousFluxTube_h_

#include "KField.h"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

class KSGenPositionHomogeneousFluxTube : public KSComponentTemplate<KSGenPositionHomogeneousFluxTube, KSGenCreator>
{
  public:
    KSGenPositionHomogeneousFluxTube();
    KSGenPositionHomogeneousFluxTube(const KSGenPositionHomogeneousFluxTube& aCopy);
    KSGenPositionHomogeneousFluxTube* Clone() const override;
    ~KSGenPositionHomogeneousFluxTube() override;

  public:
    void Dice(KSParticleQueue* aPrimaryList) override;

  public:
    void AddMagneticField(KSMagneticField* aField);

  private:
    void CalculateField(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                        KGeoBag::KThreeVector& aField);


  private:
    std::vector<KSMagneticField*> fMagneticFields;
    ;
    K_SET(double, Flux);
    ;
    K_SET(double, Rmax);
    ;
    K_SET(int, NIntegrationSteps);
    ;
    K_SET(double, Zmin);
    ;
    K_SET(double, Zmax);
    ;
    K_SET(double, Phimin);
    ;
    K_SET(double, Phimax);


  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
