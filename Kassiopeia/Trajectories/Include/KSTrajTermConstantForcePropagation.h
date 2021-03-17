#ifndef Kassiopeia_KSTrajTermConstantForcePropagation_h_
#define Kassiopeia_KSTrajTermConstantForcePropagation_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

class KSTrajTermConstantForcePropagation :
    public KSComponentTemplate<KSTrajTermConstantForcePropagation>,
    public KSTrajExactDifferentiator
{
  public:
    KSTrajTermConstantForcePropagation();
    KSTrajTermConstantForcePropagation(const KSTrajTermConstantForcePropagation& aCopy);
    KSTrajTermConstantForcePropagation* Clone() const override;
    ~KSTrajTermConstantForcePropagation() override;

  public:
    void Differentiate(double /*aTime*/, const KSTrajExactParticle& aValue,
                       KSTrajExactDerivative& aDerivative) const override;
    void SetForce(const KGeoBag::KThreeVector& aForce);

  private:
    KGeoBag::KThreeVector fForce;
};

}  // namespace Kassiopeia

#endif  // Kassiopeia_KSTrajTermConstantForcePropagation_h_
