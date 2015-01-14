#include "KGCylinder.hh"

namespace KGeoBag
{
  KGCylinder::KGCylinder(const KThreeVector& p0,
			 const KThreeVector& p1,
			 double radius) : fAxialMeshCount(0),
					  fLongitudinalMeshCount(0),
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
    KGCylinder::Visitor* tCylinderVisitor = dynamic_cast<KGCylinder::Visitor*>(aVisitor);
    if(tCylinderVisitor != NULL)
    {
      tCylinderVisitor->VisitCylinder(this);
    }
    return;
  }
  bool KGCylinder::AreaAbove(const KThreeVector& P) const
  {
    double r = ((P-fP0)*(1. - (P-fP0).Dot((fP1-fP0).Unit()))).Magnitude();

    if (r < fRadius)
      return false;
    else
      return true;
  }
  KThreeVector KGCylinder::AreaPoint(const KThreeVector& P) const
  {
    double u = (P-fP0).Dot((fP1-fP0).Unit());

    if (u<=0.)
      return fP0;
    else if (u>=1.)
      return fP1;
    else
      return fP0 + u*(fP1-fP0);
  }
  KThreeVector KGCylinder::AreaNormal(const KThreeVector& P) const
  {
    return ((P-fP0)*(1. - (P-fP0).Dot((fP1-fP0).Unit()))).Unit();
  }

}
