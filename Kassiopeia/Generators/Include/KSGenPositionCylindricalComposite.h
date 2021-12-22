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
    void SetOrigin(const katrin::KThreeVector& anOrigin);
    void SetXAxis(const katrin::KThreeVector& anXAxis);
    void SetYAxis(const katrin::KThreeVector& anYAxis);
    void SetZAxis(const katrin::KThreeVector& anZAxis);

    void SetRValue(KSGenValue* anRValue);
    void ClearRValue(KSGenValue* anRValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

  private:
    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;

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
