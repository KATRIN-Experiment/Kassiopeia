#include "KGCutConeTubeSpace.hh"

using katrin::KTwoVector;

namespace KGeoBag
{

KGCutConeTubeSpace::Visitor::Visitor() = default;
KGCutConeTubeSpace::Visitor::~Visitor() = default;

KGCutConeTubeSpace::KGCutConeTubeSpace() :
    fZ1(0.),
    fZ2(0.),
    fR11(0.),
    fR12(0.),
    fR21(0.),
    fR22(0.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fAxialMeshCount(64)
{}
KGCutConeTubeSpace::~KGCutConeTubeSpace() = default;

void KGCutConeTubeSpace::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCutConeTubeSpace::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCutConeTubeSpace::R11(const double& anR11)
{
    fR11 = anR11;
    return;
}
void KGCutConeTubeSpace::R12(const double& anR12)
{
    fR12 = anR12;
    return;
}
void KGCutConeTubeSpace::R21(const double& anR21)
{
    fR21 = anR21;
    return;
}
void KGCutConeTubeSpace::R22(const double& anR22)
{
    fR22 = anR22;
    return;
}
void KGCutConeTubeSpace::RadialMeshCount(const unsigned int& aCount)
{
    fRadialMeshCount = aCount;
    return;
}
void KGCutConeTubeSpace::RadialMeshPower(const double& aPower)
{
    fRadialMeshPower = aPower;
    return;
}
void KGCutConeTubeSpace::LongitudinalMeshCount(const unsigned int& aCount)
{
    fLongitudinalMeshCount = aCount;
    return;
}
void KGCutConeTubeSpace::LongitudinalMeshPower(const double& aPower)
{
    fLongitudinalMeshPower = aPower;
    return;
}
void KGCutConeTubeSpace::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCutConeTubeSpace::Z1() const
{
    return fZ1;
}
const double& KGCutConeTubeSpace::Z2() const
{
    return fZ2;
}
const double& KGCutConeTubeSpace::R11() const
{
    return fR11;
}
const double& KGCutConeTubeSpace::R12() const
{
    return fR12;
}
const double& KGCutConeTubeSpace::R21() const
{
    return fR21;
}
const double& KGCutConeTubeSpace::R22() const
{
    return fR22;
}
const unsigned int& KGCutConeTubeSpace::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGCutConeTubeSpace::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGCutConeTubeSpace::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGCutConeTubeSpace::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGCutConeTubeSpace::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCutConeTubeSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
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

        if (fR11 < fR12) {
            tInsideBack.Y() = fR11;
            tOutsideBack.Y() = fR12;
        }
        else {
            tInsideBack.Y() = fR12;
            tOutsideBack.Y() = fR11;
        }

        if (fR21 < fR22) {
            tInsideFront.Y() = fR21;
            tOutsideFront.Y() = fR22;
        }
        else {
            tInsideFront.Y() = fR22;
            tOutsideFront.Y() = fR21;
        }
    }
    else {
        tInsideBack.X() = fZ2;
        tOutsideBack.X() = fZ2;
        tInsideFront.X() = fZ1;
        tOutsideFront.X() = fZ1;

        if (fR21 < fR22) {
            tOutsideBack.Y() = fR21;
            tInsideBack.Y() = fR22;
        }
        else {
            tOutsideBack.Y() = fR22;
            tInsideBack.Y() = fR21;
        }

        if (fR11 < fR12) {
            tOutsideFront.Y() = fR11;
            tInsideFront.Y() = fR12;
        }
        else {
            tOutsideFront.Y() = fR12;
            tInsideFront.Y() = fR11;
        }
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
void KGCutConeTubeSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tCutConeTubeSpaceVisitor = dynamic_cast<KGCutConeTubeSpace::Visitor*>(aVisitor);
    if (tCutConeTubeSpaceVisitor != nullptr) {
        tCutConeTubeSpaceVisitor->VisitCutConeTubeSpace(this);
        return;
    }
    KGRotatedPolyLoopSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
