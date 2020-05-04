#include "KGConeSpace.hh"

namespace KGeoBag
{

KGConeSpace::Visitor::Visitor() {}
KGConeSpace::Visitor::~Visitor() {}

KGConeSpace::KGConeSpace() :
    fZA(0.),
    fZB(0.),
    fRB(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fAxialMeshCount(64)
{}
KGConeSpace::~KGConeSpace() {}

void KGConeSpace::ZA(const double& aZA)
{
    fZA = aZA;
    return;
}
void KGConeSpace::ZB(const double& aZB)
{
    fZB = aZB;
    return;
}
void KGConeSpace::RB(const double& anRB)
{
    fRB = anRB;
    return;
}
void KGConeSpace::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGConeSpace::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGConeSpace::RadialMeshCount(const unsigned int& aRadialMeshCount)
{
    fRadialMeshCount = aRadialMeshCount;
    return;
}
void KGConeSpace::RadialMeshPower(const double& aRadialMeshPower)
{
    fRadialMeshPower = aRadialMeshPower;
    return;
}
void KGConeSpace::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGConeSpace::ZA() const
{
    return fZA;
}
const double& KGConeSpace::ZB() const
{
    return fZB;
}
const double& KGConeSpace::RB() const
{
    return fRB;
}
const unsigned int& KGConeSpace::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGConeSpace::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGConeSpace::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGConeSpace::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGConeSpace::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGConeSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
{
    if (fZA < fZB) {
        fPath->X1(fZB);
        fPath->Y1(fRB);
        fPath->X2(fZA);
        fPath->Y2(0.);
    }
    else {
        fPath->X1(fZA);
        fPath->Y1(0.);
        fPath->X2(fZB);
        fPath->Y2(fRB);
    }

    fPath->MeshCount(fLongitudinalMeshCount);
    fPath->MeshPower(fLongitudinalMeshPower);
    fFlattenedMeshCount = fRadialMeshCount;
    fFlattenedMeshPower = fRadialMeshPower;
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSpace::VolumeInitialize(aBoundaryContainer);
    return;
}
void KGConeSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tConeSpaceVisitor = dynamic_cast<KGConeSpace::Visitor*>(aVisitor);
    if (tConeSpaceVisitor != nullptr) {
        tConeSpaceVisitor->VisitConeSpace(this);
        return;
    }
    KGRotatedLineSegmentSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
