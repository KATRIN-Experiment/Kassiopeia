#include "KSTrajMagneticDerivative.h"

namespace Kassiopeia
{

KSTrajMagneticDerivative::KSTrajMagneticDerivative() = default;
KSTrajMagneticDerivative::KSTrajMagneticDerivative(const KSTrajMagneticDerivative&) = default;
KSTrajMagneticDerivative::~KSTrajMagneticDerivative() = default;

void KSTrajMagneticDerivative::AddToTime(const double& aTime)
{
    fData[0] = aTime;
    return;
}
void KSTrajMagneticDerivative::AddToSpeed(const double& aSpeed)
{
    fData[1] = aSpeed;
    return;
}
void KSTrajMagneticDerivative::AddToVelocity(const KGeoBag::KThreeVector& aVelocity)
{
    fData[2] = aVelocity.X();
    fData[3] = aVelocity.Y();
    fData[4] = aVelocity.Z();
    return;
}


}  // namespace Kassiopeia
