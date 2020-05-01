#ifndef Kassiopeia_KSGenPositionSphericalComposite_h_
#define Kassiopeia_KSGenPositionSphericalComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenPositionSphericalComposite : public KSComponentTemplate<KSGenPositionSphericalComposite, KSGenCreator>
{
  public:
    KSGenPositionSphericalComposite();
    KSGenPositionSphericalComposite(const KSGenPositionSphericalComposite& aCopy);
    KSGenPositionSphericalComposite* Clone() const override;
    ~KSGenPositionSphericalComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaryList) override;

  public:
    void SetOrigin(const KThreeVector& anOrigin);
    void SetXAxis(const KThreeVector& anXAxis);
    void SetYAxis(const KThreeVector& anYAxis);
    void SetZAxis(const KThreeVector& anZAxis);

    void SetRValue(KSGenValue* anRValue);
    void ClearRValue(KSGenValue* anRValue);

    void SetThetaValue(KSGenValue* aThetaValue);
    void ClearThetaValue(KSGenValue* aThetaValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

  private:
    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;

    typedef enum
    {
        eRadius,
        eTheta,
        ePhi
    } CoordinateType;

    std::map<CoordinateType, int> fCoordinateMap;
    std::vector<std::pair<CoordinateType, KSGenValue*>> fValues;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
