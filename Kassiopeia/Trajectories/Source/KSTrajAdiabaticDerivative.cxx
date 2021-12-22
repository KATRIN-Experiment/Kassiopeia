#include "KSTrajAdiabaticDerivative.h"

using katrin::KThreeVector;

namespace Kassiopeia
{

KSTrajAdiabaticDerivative::KSTrajAdiabaticDerivative() = default;
KSTrajAdiabaticDerivative::KSTrajAdiabaticDerivative(const KSTrajAdiabaticDerivative&) = default;
KSTrajAdiabaticDerivative::~KSTrajAdiabaticDerivative() = default;

void KSTrajAdiabaticDerivative::AddToTime(const double& aTime)
{
    fData[0] += aTime;
    return;
}
void KSTrajAdiabaticDerivative::AddToSpeed(const double& aSpeed)
{
    fData[1] += aSpeed;
    return;
}
void KSTrajAdiabaticDerivative::AddToGuidingCenterVelocity(const KThreeVector& aVelocity)
{
    fData[2] += aVelocity.X();
    fData[3] += aVelocity.Y();
    fData[4] += aVelocity.Z();
    return;
}
void KSTrajAdiabaticDerivative::AddToLongitudinalForce(const double& aForce)
{
    fData[5] += aForce;
    return;
}
void KSTrajAdiabaticDerivative::AddToTransverseForce(const double& aForce)
{
    fData[6] += aForce;
    return;
}
void KSTrajAdiabaticDerivative::AddToPhaseVelocity(const double& aPhaseVelocity)
{
    fData[7] += aPhaseVelocity;
    return;
}
}  // namespace Kassiopeia
