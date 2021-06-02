#ifndef Kassiopeia_KSTrajTermPropagation_h_
#define Kassiopeia_KSTrajTermPropagation_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTrappedTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajTermPropagation :
    public KSComponentTemplate<KSTrajTermPropagation>,
    public KSTrajExactDifferentiator,
    public KSTrajExactSpinDifferentiator,
    public KSTrajExactTrappedDifferentiator,
    public KSTrajAdiabaticDifferentiator,
    public KSTrajAdiabaticSpinDifferentiator,
    public KSTrajElectricDifferentiator,
    public KSTrajMagneticDifferentiator
{
  public:
    KSTrajTermPropagation();
    KSTrajTermPropagation(const KSTrajTermPropagation& aCopy);
    KSTrajTermPropagation* Clone() const override;
    ~KSTrajTermPropagation() override;

  public:
    void Differentiate(double /*aTime*/, const KSTrajExactParticle& aValue,
                       KSTrajExactDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajExactSpinParticle& aValue,
                       KSTrajExactSpinDerivative& aDerivative) const override;
    void Differentiate(double aTime, const KSTrajExactTrappedParticle& aValue,
                       KSTrajExactTrappedDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aValue,
                       KSTrajAdiabaticDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajAdiabaticSpinParticle& aValue,
                       KSTrajAdiabaticSpinDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aValue,
                       KSTrajMagneticDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajElectricParticle& aValue,
                       KSTrajElectricDerivative& aDerivative) const override;

  public:
    typedef enum
    {
        eBackward = -1,
        eForward = 1
    } Direction;

    void SetDirection(const Direction& anDirection);
    void ReverseDirection() { fDirection = (fDirection == Direction::eBackward ? Direction::eForward : Direction::eBackward); }

  private:
    Direction fDirection;
};

}  // namespace Kassiopeia

#endif
