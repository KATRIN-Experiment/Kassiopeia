#include "KSTrajExactTrappedDerivative.h"

using katrin::KThreeVector;

namespace Kassiopeia
{

KSTrajExactTrappedDerivative::KSTrajExactTrappedDerivative() = default;
KSTrajExactTrappedDerivative::KSTrajExactTrappedDerivative(const KSTrajExactTrappedDerivative&) = default;
KSTrajExactTrappedDerivative::~KSTrajExactTrappedDerivative() = default;

void KSTrajExactTrappedDerivative::AddToTime(const double& aTime)
{
    fData[0] += aTime;
    return;
}
void KSTrajExactTrappedDerivative::AddToSpeed(const double& aSpeed)
{
    fData[1] += aSpeed;
    return;
}
void KSTrajExactTrappedDerivative::AddToVelocity(const KThreeVector& aVelocity)
{
    fData[2] += aVelocity.X();
    fData[3] += aVelocity.Y();
    fData[4] += aVelocity.Z();
    return;
}
void KSTrajExactTrappedDerivative::AddToForce(const KThreeVector& aForce)
{
    fData[5] += aForce.X();
    fData[6] += aForce.Y();
    fData[7] += aForce.Z();
    return;
}
}  // namespace Kassiopeia
