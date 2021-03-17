#ifndef KSGENMOMENTUMRECTANGULARCOMPOSITE_H
#define KSGENMOMENTUMRECTANGULARCOMPOSITE_H

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenMomentumRectangularComposite : public KSComponentTemplate<KSGenMomentumRectangularComposite, KSGenCreator>
{
  public:
    KSGenMomentumRectangularComposite();
    KSGenMomentumRectangularComposite(const KSGenMomentumRectangularComposite& aCopy);
    KSGenMomentumRectangularComposite* Clone() const override;
    ~KSGenMomentumRectangularComposite() override;

  public:
    void Dice(KSParticleQueue* aParticleList) override;

  public:
    void SetXAxis(const KGeoBag::KThreeVector& anXAxis);
    void SetYAxis(const KGeoBag::KThreeVector& anYAxis);
    void SetZAxis(const KGeoBag::KThreeVector& anZAxis);

    void SetXValue(KSGenValue* anXValue);
    void ClearXValue(KSGenValue* anXValue);
    void SetYValue(KSGenValue* aYValue);
    void ClearYValue(KSGenValue* aYValue);
    void SetZValue(KSGenValue* aZValue);
    void ClearZValue(KSGenValue* aZValue);


  private:
    KSGenValue* fXValue;
    KSGenValue* fYValue;
    KSGenValue* fZValue;

    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif  // KSGENMOMENTUMRECTANGULARCOMPOSITE_H
