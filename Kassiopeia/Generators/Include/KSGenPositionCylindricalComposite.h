#ifndef Kassiopeia_KSGenPositionCylindricalComposite_h_
#define Kassiopeia_KSGenPositionCylindricalComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

#include <utility>

namespace Kassiopeia
{


class KSGenPositionCylindricalComposite : public KSComponentTemplate<KSGenPositionCylindricalComposite, KSGenCreator>
{
  public:
    KSGenPositionCylindricalComposite();
    KSGenPositionCylindricalComposite(const KSGenPositionCylindricalComposite& aCopy);
    KSGenPositionCylindricalComposite* Clone() const override;
    ~KSGenPositionCylindricalComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaryList) override;

  public:
    void SetOrigin(const KGeoBag::KThreeVector& anOrigin);
    void SetXAxis(const KGeoBag::KThreeVector& anXAxis);
    void SetYAxis(const KGeoBag::KThreeVector& anYAxis);
    void SetZAxis(const KGeoBag::KThreeVector& anZAxis);

    void SetRValue(KSGenValue* anRValue);
    void ClearRValue(KSGenValue* anRValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

  private:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    typedef enum
    {
        eRadius,
        ePhi,
        eZ
    } CoordinateType;

    std::map<CoordinateType, int> fCoordinateMap;
    std::vector<std::pair<CoordinateType, KSGenValue*>> fValues;


  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
