#include "KGCylinderSpace.hh"

namespace KGeoBag
{

KGCylinderSpace::Visitor::Visitor() {}
KGCylinderSpace::Visitor::~Visitor() {}

KGCylinderSpace::KGCylinderSpace() :
    fZ1(0.),
    fZ2(0.),
    fR(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fAxialMeshCount(64)
{}
KGCylinderSpace::~KGCylinderSpace() {}

void KGCylinderSpace::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCylinderSpace::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCylinderSpace::R(const double& anR)
{
    fR = anR;
    return;
}
void KGCylinderSpace::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGCylinderSpace::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGCylinderSpace::RadialMeshCount(const unsigned int& aRadialMeshCount)
{
    fRadialMeshCount = aRadialMeshCount;
    return;
}
void KGCylinderSpace::RadialMeshPower(const double& aRadialMeshPower)
{
    fRadialMeshPower = aRadialMeshPower;
    return;
}
void KGCylinderSpace::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCylinderSpace::Z1() const
{
    return fZ1;
}
const double& KGCylinderSpace::Z2() const
{
    return fZ2;
}
const double& KGCylinderSpace::R() const
{
    return fR;
}
const unsigned int& KGCylinderSpace::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGCylinderSpace::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGCylinderSpace::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGCylinderSpace::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGCylinderSpace::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCylinderSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
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
    fFlattenedMeshCount = fRadialMeshCount;
    fFlattenedMeshPower = fRadialMeshPower;
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSpace::VolumeInitialize(aBoundaryContainer);
    return;
}
void KGCylinderSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tCylinderSpaceVisitor = dynamic_cast<KGCylinderSpace::Visitor*>(aVisitor);
    if (tCylinderSpaceVisitor != nullptr) {
        tCylinderSpaceVisitor->VisitCylinderSpace(this);
        return;
    }
    KGRotatedLineSegmentSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
