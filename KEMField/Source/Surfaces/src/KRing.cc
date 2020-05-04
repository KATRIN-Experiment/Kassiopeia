#include "../../../Surfaces/include/KRing.hh"

#include "../../../Surfaces/include/KSurfaceVisitors.hh"
#include "KEMConstants.hh"

namespace KEMField
{
void KRing::SetValues(const KPosition& p)
{
    fP = p;
}

void KRing::SetValues(const double& r, const double& z)
{
    fP[0] = r;
    fP[1] = 0.;
    fP[2] = z;
}

double KRing::Area() const
{
    return 2. * KEMConstants::Pi * fP[0];
}

double KRing::DistanceTo(const KPosition& aPoint, KPosition& nearestPoint)
{
    double r = sqrt(aPoint[0] * aPoint[0] + aPoint[1] * aPoint[1]);

    double cos = aPoint[0] / sqrt(aPoint[0] * aPoint[0] + aPoint[1] * aPoint[1]);
    double sin = aPoint[1] / sqrt(aPoint[0] * aPoint[0] + aPoint[1] * aPoint[1]);

    nearestPoint[0] = fP[0] * cos;
    nearestPoint[1] = fP[0] * sin;
    nearestPoint[2] = fP[2];

    return sqrt((fP[0] - r) * (fP[0] - r) + (aPoint[2] - fP[2]) * (aPoint[2] - fP[2]));
}

const KDirection KRing::Normal() const
{
    return KDirection(1., 0., 0.);
}
}  // namespace KEMField
