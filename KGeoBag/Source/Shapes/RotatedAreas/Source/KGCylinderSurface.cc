#include "KGCylinderSurface.hh"

namespace KGeoBag
{

KGCylinderSurface::Visitor::Visitor() = default;
KGCylinderSurface::Visitor::~Visitor() = default;

KGCylinderSurface::KGCylinderSurface() :
    fZ1(0.),
    fZ2(0.),
    fR(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fAxialMeshCount(8)
{}
KGCylinderSurface::~KGCylinderSurface() = default;

void KGCylinderSurface::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCylinderSurface::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCylinderSurface::R(const double& anR)
{
    fR = anR;
    return;
}
void KGCylinderSurface::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGCylinderSurface::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGCylinderSurface::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCylinderSurface::Z1() const
{
    return fZ1;
}
const double& KGCylinderSurface::Z2() const
{
    return fZ2;
}
const double& KGCylinderSurface::R() const
{
    return fR;
}
const unsigned int& KGCylinderSurface::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGCylinderSurface::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGCylinderSurface::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCylinderSurface::AreaInitialize() const
{
    if (fZ1 < fZ2) {
        fPath->X1(fZ2);
        fPath->Y1(fR);
        fPath->X2(fZ1);
        fPath->Y2(fR);
    }
    else {
        fPath->X1(fZ1);
        fPath->Y1(fR);
        fPath->X2(fZ2);
        fPath->Y2(fR);
    }

    fPath->MeshCount(fLongitudinalMeshCount);
    fPath->MeshPower(fLongitudinalMeshPower);
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSurface::AreaInitialize();
    return;
}
void KGCylinderSurface::AreaAccept(KGVisitor* aVisitor)
{
    auto* tCylinderSurfaceVisitor = dynamic_cast<KGCylinderSurface::Visitor*>(aVisitor);
    if (tCylinderSurfaceVisitor != nullptr) {
        tCylinderSurfaceVisitor->VisitCylinderSurface(this);
        return;
    }
    KGRotatedLineSegmentSurface::AreaAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
