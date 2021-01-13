#include "KGConeSurface.hh"

namespace KGeoBag
{

KGConeSurface::Visitor::Visitor() = default;
KGConeSurface::Visitor::~Visitor() = default;

KGConeSurface::KGConeSurface() :
    fZA(0.),
    fZB(0.),
    fRB(0.),
    fLongitudinalMeshCount(8),
    fLongitudinalMeshPower(1.),
    fAxialMeshCount(8)
{}
KGConeSurface::~KGConeSurface() = default;

void KGConeSurface::ZA(const double& aZA)
{
    fZA = aZA;
    return;
}
void KGConeSurface::ZB(const double& aZB)
{
    fZB = aZB;
    return;
}
void KGConeSurface::RB(const double& anRB)
{
    fRB = anRB;
    return;
}
void KGConeSurface::LongitudinalMeshCount(const unsigned int& aLongitudinalMeshCount)
{
    fLongitudinalMeshCount = aLongitudinalMeshCount;
    return;
}
void KGConeSurface::LongitudinalMeshPower(const double& aLongitudinalMeshPower)
{
    fLongitudinalMeshPower = aLongitudinalMeshPower;
    return;
}
void KGConeSurface::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGConeSurface::ZA() const
{
    return fZA;
}
const double& KGConeSurface::ZB() const
{
    return fZB;
}
const double& KGConeSurface::RB() const
{
    return fRB;
}
const unsigned int& KGConeSurface::LongitudinalMeshCount() const
{
    return fLongitudinalMeshCount;
}
const double& KGConeSurface::LongitudinalMeshPower() const
{
    return fLongitudinalMeshPower;
}
const unsigned int& KGConeSurface::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGConeSurface::AreaInitialize() const
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
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSurface::AreaInitialize();
    return;
}
void KGConeSurface::AreaAccept(KGVisitor* aVisitor)
{
    auto* tConeSurfaceVisitor = dynamic_cast<KGConeSurface::Visitor*>(aVisitor);
    if (tConeSurfaceVisitor != nullptr) {
        tConeSurfaceVisitor->VisitConeSurface(this);
    }
    KGRotatedLineSegmentSurface::AreaAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
