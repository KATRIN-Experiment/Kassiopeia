#include "KSTrajMagneticDerivative.h"

namespace Kassiopeia
{

KSTrajMagneticDerivative::KSTrajMagneticDerivative() {}
KSTrajMagneticDerivative::KSTrajMagneticDerivative(const KSTrajMagneticDerivative& anOperand) :
    KSMathArray<5>(anOperand)
{}
KSTrajMagneticDerivative::~KSTrajMagneticDerivative() {}

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
void KSTrajMagneticDerivative::AddToVelocity(const KThreeVector& aVelocity)
{
    fData[2] = aVelocity.X();
    fData[3] = aVelocity.Y();
    fData[4] = aVelocity.Z();
    return;
}


}  // namespace Kassiopeia
