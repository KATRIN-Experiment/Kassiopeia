#include "KGDisk.hh"

namespace KGeoBag
{
KGDisk::KGDisk(const KThreeVector& p0, const KThreeVector& normal, double radius)
{
    fP0 = p0;
    fNormal = normal.Unit();
    fRadius = radius;
}

//  KGDisk* KGDisk::AreaClone() const
//  {
//    KGDisk* d = new KGDisk();
//    d->fP0 = fP0;
//    d->fNormal = fNormal;
//    d->fRadius = fRadius;
//
//    return d;
//  }

void KGDisk::AreaAccept(KGVisitor* aVisitor)
{
    auto* tDiskVisitor = dynamic_cast<KGDisk::Visitor*>(aVisitor);
    if (tDiskVisitor != nullptr) {
        tDiskVisitor->Visit(this);
    }
    return;
}

bool KGDisk::AreaAbove(const KThreeVector& aPoint) const
{
    if ((aPoint - fP0).Dot(fNormal) > 0.)
        return true;
    else
        return false;
}

KThreeVector KGDisk::AreaPoint(const KThreeVector& aPoint) const
{
    double mag = (aPoint - fP0).Dot(fNormal);
    KThreeVector unit = ((aPoint - fP0) - fNormal * mag).Unit();

    if (mag < -fRadius)
        mag = -fRadius;
    else if (mag > fRadius)
        mag = fRadius;

    return fP0 + mag * unit;
}

KThreeVector KGDisk::AreaNormal(const KThreeVector& aPoint) const
{
    if ((aPoint - fP0).Dot(fNormal) > 0.)
        return fNormal;
    else
        return -1. * fNormal;
}
}  // namespace KGeoBag
