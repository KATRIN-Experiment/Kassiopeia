#include "KGTriangle.hh"

namespace KGeoBag
{

KGTriangle::KGTriangle(const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1,
                         const KThreeVector& n2) :
    fA(a),
    fB(b),
    fP0(p0),
    fN1(n1),
    fN2(n2)
{}

KGTriangle::KGTriangle(const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& p2)
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p2 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
}

KGTriangle::KGTriangle(const KGTriangle& t) :
    KGTriangle(t.fA, t.fB, t.fP0, t.fN1, t.fN2)
{}

KGTriangle& KGTriangle::operator=(const KGTriangle& t)
{
    if (this == &t)
        return *this;

    fA = t.fA;
    fB = t.fB;
    fP0 = t.fP0;
    fN1 = t.fN1;
    fN2 = t.fN2;
    return *this;
}

void KGTriangle::AreaAccept(KGVisitor* aVisitor)
{
    auto* tTriangleVisitor = dynamic_cast<KGTriangle::Visitor*>(aVisitor);
    if (tTriangleVisitor != nullptr) {
        tTriangleVisitor->Visit(this);
    }
    return;
}

bool KGTriangle::AreaAbove(const KThreeVector& aPoint) const
{
    if ((aPoint - fP0).Dot(GetN3()) > 0.)
        return true;
    else
        return false;
}

KThreeVector KGTriangle::AreaPoint(const KThreeVector& aPoint) const
{
    /// TODO

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

    return fP0 + u * fN1 + v * fN2;
}

KThreeVector KGTriangle::AreaNormal(const KThreeVector& aPoint) const
{
    KThreeVector n3 = GetN3();
    if ((aPoint - fP0).Dot(n3) > 0.)
        return n3;
    else
        return -1. * n3;
}

bool KGTriangle::ContainsPoint(const KThreeVector& aPoint) const
{
    KThreeVector p0 = GetP0();
    KThreeVector p1 = GetP1();
    KThreeVector p2 = GetP2();
    if (SameSide(aPoint, p0, p1, p2) && SameSide(aPoint, p1, p0, p2) && SameSide(aPoint, p2, p0, p1))
        return true;
    else
        return false;
}

/**
   * Returns the minimum distance between the triangle and a point P.
   * This code is adapted from: "Distance Between Point and Triangle in 3D",
   * David Eberly, Geometric Tools, LLC (http://www.geometrictools.com/).
   */
double KGTriangle::DistanceTo(const KThreeVector& aPoint, KThreeVector& nearestPoint)
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

bool KGTriangle::SameSide(const KThreeVector& point, const KThreeVector& A, const KThreeVector& B,
                              const KThreeVector& C)
{
    KThreeVector cp1 = (B - A).Cross(point - A);
    KThreeVector cp2 = (B - A).Cross(C - A);
    if (cp1.Dot(cp2) > 0) {
        return true;
    }
    return false;
}

}
