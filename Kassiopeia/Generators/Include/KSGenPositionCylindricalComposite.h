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
    void SetOrigin(const KThreeVector& anOrigin);
    void SetXAxis(const KThreeVector& anXAxis);
    void SetYAxis(const KThreeVector& anYAxis);
    void SetZAxis(const KThreeVector& anZAxis);

    void SetRValue(KSGenValue* anRValue);
    void ClearRValue(KSGenValue* anRValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

  private:
    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;

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
