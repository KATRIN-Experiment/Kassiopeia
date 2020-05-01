#include "KGRectangle.hh"

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
}  // namespace KGeoBag
