#include "KLineSegment.hh"

#include <cmath>

namespace KEMField
{
void KLineSegment::SetValues(const KPosition& p0, const KPosition& p1, const double& diameter)
{
    fP0 = p0;
    fP1 = p1;
    fDiameter = diameter;
}

double KLineSegment::DistanceTo(const KPosition& aPoint, KPosition& nearestPoint)
{
    double u = (aPoint - fP0) * (fP1 - fP0) / (fP1 - fP0).MagnitudeSquared();

    if (u <= 0.)
        nearestPoint = fP0;
    else if (u >= 1.)
        nearestPoint = fP1;
    else
        nearestPoint = fP0 + u * (fP1 - fP0);
    return (aPoint - nearestPoint).Magnitude();
}

const KDirection KLineSegment::Normal() const
{
    static const KDirection x(1., 0., 0.);
    static const KDirection y(0., 1., 0.);
    KDirection d(fP1 - fP0);
    return (d.Dot(x) > 1.e-10 ? d.Cross(x).Unit() : d.Cross(y).Unit());
}
}  // namespace KEMField
