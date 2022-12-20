#include "KSTrajTermGravity.h"

#include "KConst.h"

namespace Kassiopeia
{

KSTrajTermGravity::KSTrajTermGravity() : fGravity(katrin::KThreeVector(0., 0., 0.)) {}
KSTrajTermGravity::KSTrajTermGravity(const KSTrajTermGravity& aCopy) : KSComponent(aCopy), fGravity(aCopy.fGravity) {}
KSTrajTermGravity* KSTrajTermGravity::Clone() const
{
    return new KSTrajTermGravity(*this);
}
KSTrajTermGravity::~KSTrajTermGravity() = default;

void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle,
                                      KSTrajExactDerivative& aDerivative) const
{
    aDerivative.AddToForce(fGravity * aParticle.GetMass());
}

void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajExactSpinParticle& aParticle,
                                      KSTrajExactSpinDerivative& aDerivative) const
{
    aDerivative.AddToForce(fGravity * aParticle.GetMass());
}

// void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const
// {
//     aDerivative.AddToForce( fGravity );
// }

void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajAdiabaticSpinParticle& aParticle,
                                      KSTrajAdiabaticSpinDerivative& aDerivative) const
{
    aDerivative.AddToForce(fGravity * aParticle.GetMass());
}

// void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajElectricParticle& aParticle, KSTrajElectricDerivative& aDerivative ) const
// {
//     aDerivative.AddToForce( fGravity );
// }
//
// void KSTrajTermGravity::Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aParticle, KSTrajMagneticDerivative& aDerivative ) const
// {
//     aDerivative.AddToForce( fGravity );
// }

void KSTrajTermGravity::SetGravity(const katrin::KThreeVector& aGravity)
{
    fGravity = aGravity;
    return;
}

}  // namespace Kassiopeia
