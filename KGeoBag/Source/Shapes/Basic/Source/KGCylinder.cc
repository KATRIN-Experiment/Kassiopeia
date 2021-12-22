#include "KGCylinder.hh"

namespace KGeoBag
{
KGCylinder::KGCylinder(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, double radius) :
    fAxialMeshCount(8),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.)
{
    fP0 = p0;
    fP1 = p1;
    fRadius = radius;
}

//  KGCylinder* KGCylinder::AreaClone() const
//  {
//    KGCylinder* c = new KGCylinder();
//    c->fP0 = fP0;
//    c->fP1 = fP1;
//    c->fRadius = fRadius;
//    c->fAxialMeshCount = fAxialMeshCount;
//    c->fLongitudinalMeshCount = fLongitudinalMeshCount;
//    c->fLongitudinalMeshPower = fLongitudinalMeshPower;
//
//    return c;
//  }

void KGCylinder::AreaAccept(KGVisitor* aVisitor)
{
    auto* tCylinderVisitor = dynamic_cast<KGCylinder::Visitor*>(aVisitor);
    if (tCylinderVisitor != nullptr) {
        tCylinderVisitor->VisitCylinder(this);
    }
    return;
}
bool KGCylinder::AreaAbove(const katrin::KThreeVector& P) const
{
    double r = ((P - fP0) * (1. - (P - fP0).Dot((fP1 - fP0).Unit()))).Magnitude();

    if (r < fRadius)
        return false;
    else
        return true;
}
katrin::KThreeVector KGCylinder::AreaPoint(const katrin::KThreeVector& P) const
{
    double u = (P - fP0).Dot((fP1 - fP0).Unit());

    if (u <= 0.)
        return fP0;
    else if (u >= 1.)
        return fP1;
    else
        return fP0 + u * (fP1 - fP0);
}
katrin::KThreeVector KGCylinder::AreaNormal(const katrin::KThreeVector& P) const
{
    return ((P - fP0) * (1. - (P - fP0).Dot((fP1 - fP0).Unit()))).Unit();
}

}  // namespace KGeoBag
