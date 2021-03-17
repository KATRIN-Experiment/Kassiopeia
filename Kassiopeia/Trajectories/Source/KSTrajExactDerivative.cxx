#include "KSTrajExactDerivative.h"

namespace Kassiopeia
{

KSTrajExactDerivative::KSTrajExactDerivative() = default;
KSTrajExactDerivative::KSTrajExactDerivative(const KSTrajExactDerivative&) = default;
KSTrajExactDerivative::~KSTrajExactDerivative() = default;

void KSTrajExactDerivative::AddToTime(const double& aTime)
{
    fData[0] += aTime;
    return;
}
void KSTrajExactDerivative::AddToSpeed(const double& aSpeed)
{
    fData[1] += aSpeed;
    return;
}
void KSTrajExactDerivative::AddToVelocity(const KGeoBag::KThreeVector& aVelocity)
{
    fData[2] += aVelocity.X();
    fData[3] += aVelocity.Y();
    fData[4] += aVelocity.Z();
    return;
}
void KSTrajExactDerivative::AddToForce(const KGeoBag::KThreeVector& aForce)
{
    fData[5] += aForce.X();
    fData[6] += aForce.Y();
    fData[7] += aForce.Z();
    return;
}
}  // namespace Kassiopeia
