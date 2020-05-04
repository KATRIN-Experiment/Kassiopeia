#include "KGCutConeSpace.hh"

namespace KGeoBag
{

KGCutConeSpace::Visitor::Visitor() {}
KGCutConeSpace::Visitor::~Visitor() {}

KGCutConeSpace::KGCutConeSpace() :
    fZ1(0.),
    fR1(0.),
    fZ2(0.),
    fR2(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fAxialMeshCount(64)
{}
KGCutConeSpace::~KGCutConeSpace() {}

void KGCutConeSpace::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCutConeSpace::R1(const double& anR1)
{
    fR1 = anR1;
    return;
}
void KGCutConeSpace::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCutConeSpace::R2(const double& anR2)
{
    fR2 = anR2;
    return;
}
void KGCutConeSpace::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGCutConeSpace::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGCutConeSpace::RadialMeshCount(const unsigned int& aRadialMeshCount)
{
    fRadialMeshCount = aRadialMeshCount;
    return;
}
void KGCutConeSpace::RadialMeshPower(const double& aRadialMeshPower)
{
    fRadialMeshPower = aRadialMeshPower;
    return;
}
void KGCutConeSpace::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCutConeSpace::Z1() const
{
    return fZ1;
}
const double& KGCutConeSpace::R1() const
{
    return fR1;
}
const double& KGCutConeSpace::Z2() const
{
    return fZ2;
}
const double& KGCutConeSpace::R2() const
{
    return fR2;
}
const unsigned int& KGCutConeSpace::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGCutConeSpace::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGCutConeSpace::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGCutConeSpace::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGCutConeSpace::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCutConeSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
{
    if (fZ1 < fZ2) {
        fPath->X1(fZ2);
        fPath->Y1(fR2);
        fPath->X2(fZ1);
        fPath->Y2(fR1);
    }
    else {
        fPath->X1(fZ1);
        fPath->Y1(fR1);
        fPath->X2(fZ2);
        fPath->Y2(fR2);
    }

    fPath->MeshCount(fLongitudinalMeshCount);
    fPath->MeshPower(fLongitudinalMeshPower);
    fFlattenedMeshCount = fRadialMeshCount;
    fFlattenedMeshPower = fRadialMeshPower;
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSpace::VolumeInitialize(aBoundaryContainer);
    return;
}
void KGCutConeSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tCutConeSpaceVisitor = dynamic_cast<KGCutConeSpace::Visitor*>(aVisitor);
    if (tCutConeSpaceVisitor != nullptr) {
        tCutConeSpaceVisitor->VisitCutConeSpace(this);
        return;
    }
    KGRotatedLineSegmentSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
