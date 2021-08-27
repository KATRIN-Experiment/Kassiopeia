#include "KTriangle.hh"

#include "KSurfaceVisitors.hh"

namespace KEMField
{
void KTriangle::SetValues(const double& a, const double& b, const KPosition& p0, const KDirection& n1,
                          const KDirection& n2)
{
    fA = a;
    fB = b;
    fP0 = p0;
    fN1 = n1;
    fN2 = n2;
    fN3 = n1.Cross(n2).Unit();
}

void KTriangle::SetValues(const KPosition& p0, const KPosition& p1, const KPosition& p2)
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p2 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
    fN3 = fN1.Cross(fN2).Unit();
}

/**
   * Returns the minimum distance between the triangle and a point P.
   * This code is adapted from: "Distance Between Point and Triangle in 3D",
   * David Eberly, Geometric Tools, LLC (http://www.geometrictools.com/).
   */
double KTriangle::DistanceTo(const KPosition& aPoint, KPosition& nearestPoint)
{
    double a = fA * fA;
    double b = fA * fB * fN1.Dot(fN2);
    double c = fB * fB;
    double d = fA * fN1.Dot(fP0 - aPoint);
    double e = fB * fN2.Dot(fP0 - aPoint);
    double f = (fP0 - aPoint).MagnitudeSquared();

    int region;

    double distance;

    double det = fabs(a * c - b * b);
    double s = b * e - c * d;
    double t = b * d - a * e;

    if ((s + t) <= det) {
        if (s < 0) {
            if (t < 0)
                region = 4;
            else
                region = 3;
        }
        else if (t < 0)
            region = 5;
        else
            region = 0;
    }
    else {
        if (s < 0)
            region = 2;
        else if (t < 0)
            region = 6;
        else
            region = 1;
    }

    switch (region) {
        case 0: {
            double invDet = 1. / det;
            s *= invDet;
            t *= invDet;
            double val = s * (a * s + b * t + 2. * d) + t * (b * s + c * t + 2. * e) + f;
            if (val < 1.e-14)
                distance = 0.;
            else
                distance = sqrt(val);
        } break;
        case 1: {
            double numerator = (c + e - b - d);
            if (numerator <= 0.) {
                s = 0.;
                t = 1.;
                distance = sqrt(c + 2. * e + f);
            }
            else {
                double denominator = a - 2. * b + c;
                if (numerator >= denominator) {
                    s = 1.;
                    t = 0.;
                    distance = sqrt(a + 2. * d + f);
                }
                else {
                    s = numerator / denominator;
                    t = 1. - s;
                    distance = sqrt(s * (a * s + b * t + 2. * d) + t * (b * s + c * t + 2. * e) + f);
                }
            }
        } break;
        case 2: {
            double tmp0 = b + d;
            double tmp1 = c + e;
            if (tmp1 > tmp0) {
                double numerator = tmp1 - tmp0;
                double denominator = a - 2. * b + c;
                if (numerator >= denominator) {
                    s = 1.;
                    t = 0.;
                    distance = sqrt(a + 2. * d + f);
                }
                else {
                    s = numerator / denominator;
                    t = 1. - s;
                    distance = sqrt(s * (a * s + b * t + 2. * d) + t * (b * s + c * t + 2. * e) + f);
                }
            }
            else {
                s = 0.;
                if (tmp1 <= 0.) {
                    t = 1.;
                    distance = sqrt(c + 2. * e + f);
                }
                else if (e >= 0) {
                    t = 0.;
                    distance = sqrt(f);
                }
                else {
                    t = -e / c;
                    distance = sqrt(e * t + f);
                }
            }
        } break;
        case 3: {
            s = 0.;
            if (e >= 0) {
                t = 0;
                distance = sqrt(f);
            }
            else if (-e >= c) {
                t = 1.;
                distance = sqrt(c + 2. * e + f);
            }
            else {
                t = -e / c;
                distance = sqrt(e * t + f);
            }
        } break;
        case 4: {
            if (d < 0) {
                t = 0.;
                if (-d >= a) {
                    s = 1.;
                    distance = sqrt(a + 2. * d + f);
                }
                else {
                    s = -d / a;
                    distance = sqrt(d * s + f);
                }
            }
            else {
                s = 0.;
                if (e >= 0.) {
                    t = 0.;
                    distance = sqrt(f);
                }
                else if (-e >= c) {
                    t = 1.;
                    distance = sqrt(c + 2. * e + f);
                }
                else {
                    t = -e / c;
                    distance = sqrt(c + 2. * e + f);
                }
            }
        } break;
        case 5: {
            t = 0.;
            if (d >= 0) {
                s = 0.;
                distance = sqrt(f);
            }
            else if (-d >= a) {
                s = 1.;
                distance = sqrt(a + 2. * d + f);
            }
            else {
                s = -d / a;
                distance = sqrt(d * s + f);
            }
        } break;
        case 6: {
            double tmp0 = b + e;
            double tmp1 = a + d;
            if (tmp1 > tmp0) {
                double numerator = tmp1 - tmp0;
                double denominator = a - 2. * b + c;
                if (numerator >= denominator) {
                    t = 1.;
                    s = 0.;
                    distance = sqrt(c + 2. * e + f);
                }
                else {
                    t = numerator / denominator;
                    s = 1. - t;
                    distance = sqrt(s * (a * s + b * t + 2. * d) + t * (b * s + c * t + 2. * e) + f);
                }
            }
            else {
                t = 0.;
                if (tmp1 <= 0.) {
                    s = 1.;
                    distance = sqrt(a + 2. * d + f);
                }
                else if (d >= 0) {
                    s = 0.;
                    distance = sqrt(f);
                }
                else {
                    s = -d / a;
                    distance = sqrt(d * s + f);
                }
            }
        } break;
    }

    nearestPoint = fP0 + fA * s * fN1 + fB * t * fN2;

    return distance;
}
}  // namespace KEMField
