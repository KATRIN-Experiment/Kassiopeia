#include "KSTrajTermConstantForcePropagation.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

KSTrajTermConstantForcePropagation::KSTrajTermConstantForcePropagation() = default;
KSTrajTermConstantForcePropagation::KSTrajTermConstantForcePropagation(const KSTrajTermConstantForcePropagation&) :
    KSComponent()
{}
KSTrajTermConstantForcePropagation* KSTrajTermConstantForcePropagation::Clone() const
{
    return new KSTrajTermConstantForcePropagation(*this);
}
KSTrajTermConstantForcePropagation::~KSTrajTermConstantForcePropagation() = default;

void KSTrajTermConstantForcePropagation::Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle,
                                                       KSTrajExactDerivative& aDerivative) const
{
    KGeoBag::KThreeVector tVelocity = aParticle.GetVelocity();

    aDerivative.AddToVelocity(tVelocity);
    aDerivative.AddToForce(fForce);

    return;
}

void KSTrajTermConstantForcePropagation::SetForce(const KGeoBag::KThreeVector& aForce)
{
    fForce = aForce;
}
}  // namespace Kassiopeia
