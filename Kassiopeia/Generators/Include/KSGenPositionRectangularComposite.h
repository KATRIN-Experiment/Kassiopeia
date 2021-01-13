#ifndef Kassiopeia_KSGenPositionRectangularComposite_h_
#define Kassiopeia_KSGenPositionRectangularComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenPositionRectangularComposite : public KSComponentTemplate<KSGenPositionRectangularComposite, KSGenCreator>
{
  public:
    KSGenPositionRectangularComposite();
    KSGenPositionRectangularComposite(const KSGenPositionRectangularComposite& aCopy);
    KSGenPositionRectangularComposite* Clone() const override;
    ~KSGenPositionRectangularComposite() override;

  public:
    void Dice(KSParticleQueue* aPrimaryList) override;

  public:
    void SetXValue(KSGenValue* anXValue);
    void ClearXValue(KSGenValue* anXValue);

    void SetYValue(KSGenValue* aYValue);
    void ClearYValue(KSGenValue* aYValue);

    void SetZValue(KSGenValue* anZValue);
    void ClearZValue(KSGenValue* anZValue);

    void SetOrigin(const KGeoBag::KThreeVector& anOrigin);
    void SetXAxis(const KGeoBag::KThreeVector& anXAxis);
    void SetYAxis(const KGeoBag::KThreeVector& anYAxis);
    void SetZAxis(const KGeoBag::KThreeVector& anZAxis);

    typedef enum
    {
        eX,
        eY,
        eZ
    } CoordinateType;

  private:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    std::map<CoordinateType, int> fCoordinateMap;
    std::vector<std::pair<CoordinateType, KSGenValue*>> fValues;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
