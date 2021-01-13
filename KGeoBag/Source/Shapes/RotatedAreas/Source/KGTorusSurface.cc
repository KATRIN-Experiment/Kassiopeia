#include "KGTorusSurface.hh"

namespace KGeoBag
{

KGTorusSurface::Visitor::Visitor() = default;
KGTorusSurface::Visitor::~Visitor() = default;

KGTorusSurface::KGTorusSurface() : fZ(0.), fR(0.), fRadius(0.), fToroidalMeshCount(64), fAxialMeshCount(64) {}
KGTorusSurface::~KGTorusSurface() = default;

void KGTorusSurface::Z(const double& aZ)
{
    fZ = aZ;
    return;
}
void KGTorusSurface::R(const double& anR)
{
    fR = anR;
    return;
}
void KGTorusSurface::Radius(const double& aRadius)
{
    fRadius = aRadius;
    return;
}
void KGTorusSurface::ToroidalMeshCount(const unsigned int& aToroidalMeshCount)
{
    fToroidalMeshCount = aToroidalMeshCount;
    return;
}
void KGTorusSurface::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGTorusSurface::Z() const
{
    return fZ;
}
const double& KGTorusSurface::R() const
{
    return fR;
}
const double& KGTorusSurface::Radius() const
{
    return fRadius;
}
const unsigned int& KGTorusSurface::ToroidalMeshCount() const
{
    return fToroidalMeshCount;
}
const unsigned int& KGTorusSurface::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGTorusSurface::AreaInitialize() const
{
    fPath->X(fZ);
    fPath->Y(fR);
    fPath->Radius(fRadius);

    fPath->MeshCount(fToroidalMeshCount);
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedCircleSurface::AreaInitialize();
    return;
}
void KGTorusSurface::AreaAccept(KGVisitor* aVisitor)
{
    auto* tTorusSurfaceVisitor = dynamic_cast<KGTorusSurface::Visitor*>(aVisitor);
    if (tTorusSurfaceVisitor != nullptr) {
        tTorusSurfaceVisitor->VisitTorusSurface(this);
        return;
    }
    KGRotatedCircleSurface::AreaAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
