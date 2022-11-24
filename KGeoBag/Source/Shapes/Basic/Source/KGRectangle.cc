#include "KGRectangle.hh"

using katrin::KThreeVector;

namespace KGeoBag
{
KGRectangle::KGRectangle(const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1,
                         const KThreeVector& n2)
{
    fA = a;
    fB = b;
    fP0 = p0;
    fN1 = n1;
    fN2 = n2;
}

KGRectangle::KGRectangle(const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& /*p2*/,
                         const KThreeVector& p3)
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p3 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
}


//  KGRectangle* KGRectangle::AreaClone() const
//  {
//    KGRectangle* r = new KGRectangle();
//    r->fA = fA;
//    r->fB = fB;
//    r->fP0 = fP0;
//    r->fN1 = fN1;
//    r->fN2 = fN2;
//
//    return r;
//  }

void KGRectangle::AreaAccept(KGVisitor* aVisitor)
{
    auto* tRectangleVisitor = dynamic_cast<KGRectangle::Visitor*>(aVisitor);
    if (tRectangleVisitor != nullptr) {
        tRectangleVisitor->Visit(this);
    }
    return;
}

bool KGRectangle::AreaAbove(const KThreeVector& aPoint) const
{
    if ((aPoint - fP0).Dot(GetN3()) > 0.)
        return true;
    else
        return false;
}

KThreeVector KGRectangle::AreaPoint(const KThreeVector& aPoint) const
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

    return fP0 + u * fN1 + v * fN2;
}

KThreeVector KGRectangle::AreaNormal(const KThreeVector& aPoint) const
{
    KThreeVector n3 = GetN3();
    if ((aPoint - fP0).Dot(n3) > 0.)
        return n3;
    else
        return -1. * n3;
}

// adapted from KGTriangle::ContainsPoint()
bool KGRectangle::ContainsPoint(const KThreeVector& aPoint) const
{
    KThreeVector p0 = GetP0();
    KThreeVector p1 = GetP1();
    KThreeVector p2 = GetP2();
    KThreeVector p3 = GetP3();
    // compare two triangles covering our rectangle
    if (SameSide(aPoint, p0, p1, p2) && SameSide(aPoint, p1, p0, p2) && SameSide(aPoint, p2, p0, p1))
        return true;
    else if (SameSide(aPoint, p2, p3, p0) && SameSide(aPoint, p3, p2, p0) && SameSide(aPoint, p0, p2, p3))
        return true;
    else
        return false;
}

/**
   * Returns the shortest distance between the rectangle and a point P.
   */
double KGRectangle::DistanceTo(const KThreeVector& aPoint, KThreeVector& nearestPoint)
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

bool KGRectangle::SameSide(const KThreeVector& point, const KThreeVector& A, const KThreeVector& B, const KThreeVector& C)
{
    KThreeVector cp1 = (B - A).Cross(point - A);
    KThreeVector cp2 = (B - A).Cross(C - A);
    if (cp1.Dot(cp2) > 0) {
        return true;
    }
    return false;
}

}  // namespace KGeoBag
