#include "KGRadialDeformation.hh"

#include <cmath>

namespace KGeoBag
{
void KGRadialDeformation::Apply(katrin::KThreeVector& point) const
{
    // RadialDeformation scales the radial component of a point by a function
    // RadialScale(theta,z).

    double theta;

    if (fabs(point[0]) < 1.e-14)
        theta = M_PI / 2.;
    else
        theta = atan(fabs(point[1] / point[0]));

    if (point[0] < 1.e-14 && point[1] > -1.e-14)
        theta = M_PI - theta;
    else if (point[0] < 1.e-14 && point[1] < 1.e-14)
        theta += M_PI;
    else if (point[0] > -1.e-14 && point[1] < 1.e-14)
        theta = 2. * M_PI - theta;

    double r = sqrt(point[0] * point[0] + point[1] * point[1]) * RadialScale(theta, point[2]);

    point[0] = r * cos(theta);
    point[1] = r * sin(theta);
}
}  // namespace KGeoBag
