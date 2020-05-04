#include "../../../Surfaces/include/KRectangle.hh"

#include <cmath>

namespace KEMField
{
void KRectangle::SetValues(const double& a, const double& b, const KPosition& p0, const KDirection& n1,
                           const KDirection& n2)
{
    fA = a;
    fB = b;
    fP0 = p0;
    fN1 = n1;
    fN2 = n2;
    fN3 = fN1.Cross(fN2);
}

void KRectangle::SetValues(const KPosition& p0, const KPosition& p1, const KPosition& /*p2*/, const KPosition& p3)
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p3 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
    fN3 = fN1.Cross(fN2);
}

/**
   * Returns the shortest distance between the rectangle and a point P.
   */
double KRectangle::DistanceTo(const KPosition& aPoint, KPosition& nearestPoint)
{
    double u = (aPoint - fP0).Dot(fN1);
    if (u < 0)
        u = 0;
    else if (u > fA)
        u = fA;

    double v = (aPoint - fP0).Dot(fN2);
    if (v < 0)
        v = 0;
    else if (v > fB)
        v = fB;

    nearestPoint = fP0 + u * fN1 + v * fN2;

    return (aPoint - nearestPoint).Magnitude();
}
}  // namespace KEMField
