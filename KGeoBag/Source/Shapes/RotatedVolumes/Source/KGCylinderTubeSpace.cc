#include "KGCylinderTubeSpace.hh"

namespace KGeoBag
{

KGCylinderTubeSpace::Visitor::Visitor() = default;
KGCylinderTubeSpace::Visitor::~Visitor() = default;

KGCylinderTubeSpace::KGCylinderTubeSpace() :
    fZ1(0.),
    fZ2(0.),
    fR1(0.),
    fR2(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fAxialMeshCount(64)
{}
KGCylinderTubeSpace::~KGCylinderTubeSpace() = default;

void KGCylinderTubeSpace::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCylinderTubeSpace::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCylinderTubeSpace::R1(const double& anR1)
{
    fR1 = anR1;
    return;
}
void KGCylinderTubeSpace::R2(const double& anR2)
{
    fR2 = anR2;
    return;
}
void KGCylinderTubeSpace::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGCylinderTubeSpace::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGCylinderTubeSpace::RadialMeshCount(const unsigned int& aRadialMeshCount)
{
    fRadialMeshCount = aRadialMeshCount;
    return;
}
void KGCylinderTubeSpace::RadialMeshPower(const double& aRadialMeshPower)
{
    fRadialMeshPower = aRadialMeshPower;
    return;
}
void KGCylinderTubeSpace::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCylinderTubeSpace::Z1() const
{
    return fZ1;
}
const double& KGCylinderTubeSpace::Z2() const
{
    return fZ2;
}
const double& KGCylinderTubeSpace::R1() const
{
    return fR1;
}
const double& KGCylinderTubeSpace::R2() const
{
    return fR2;
}
const unsigned int& KGCylinderTubeSpace::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGCylinderTubeSpace::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGCylinderTubeSpace::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGCylinderTubeSpace::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGCylinderTubeSpace::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCylinderTubeSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
{
    KTwoVector tInsideBack;
    KTwoVector tInsideFront;
    KTwoVector tOutsideBack;
    KTwoVector tOutsideFront;

    if (fZ1 < fZ2) {
        tInsideBack.X() = fZ1;
        tOutsideBack.X() = fZ1;
        tInsideFront.X() = fZ2;
        tOutsideFront.X() = fZ2;
    }
    else {
        tInsideBack.X() = fZ2;
        tOutsideBack.X() = fZ2;
        tInsideFront.X() = fZ1;
        tOutsideFront.X() = fZ1;
    }

    if (fR1 < fR2) {
        tInsideBack.Y() = fR1;
        tInsideFront.Y() = fR1;
        tOutsideBack.Y() = fR2;
        tOutsideFront.Y() = fR2;
    }
    else {
        tInsideBack.Y() = fR2;
        tInsideFront.Y() = fR2;
        tOutsideBack.Y() = fR1;
        tOutsideFront.Y() = fR1;
    }

    fPath->StartPoint(tInsideBack);
    fPath->NextLine(tInsideFront, fLongitudinalMeshCount, fLongitudinalMeshPower);
    fPath->NextLine(tOutsideFront, fRadialMeshCount, fRadialMeshPower);
    fPath->NextLine(tOutsideBack, fLongitudinalMeshCount, fLongitudinalMeshPower);
    fPath->LastLine(fRadialMeshCount, fRadialMeshPower);

    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedPolyLoopSpace::VolumeInitialize(aBoundaryContainer);
    return;
}
void KGCylinderTubeSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tCylinderTubeSpaceVisitor = dynamic_cast<KGCylinderTubeSpace::Visitor*>(aVisitor);
    if (tCylinderTubeSpaceVisitor != nullptr) {
        tCylinderTubeSpaceVisitor->VisitCylinderTubeSpace(this);
        return;
    }
    KGRotatedPolyLoopSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
