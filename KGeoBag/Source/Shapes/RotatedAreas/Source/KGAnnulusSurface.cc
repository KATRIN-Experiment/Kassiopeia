#include "KGAnnulusSurface.hh"

namespace KGeoBag
{

KGAnnulusSurface::Visitor::Visitor() {}
KGAnnulusSurface::Visitor::~Visitor() {}

KGAnnulusSurface::KGAnnulusSurface() :
    fZ(0.),
    fR1(0.),
    fR2(0.),
    fRadialMeshCount(8),
    fRadialMeshPower(1.),
    fAxialMeshCount(8)
{}
KGAnnulusSurface::~KGAnnulusSurface() {}

void KGAnnulusSurface::Z(const double& aZ)
{
    fZ = aZ;
    return;
}
void KGAnnulusSurface::R1(const double& anR)
{
    fR1 = anR;
    return;
}
void KGAnnulusSurface::R2(const double& anR)
{
    fR2 = anR;
    return;
}
void KGAnnulusSurface::RadialMeshCount(const unsigned int& aRadialMeshCount)
{
    fRadialMeshCount = aRadialMeshCount;
    return;
}
void KGAnnulusSurface::RadialMeshPower(const double& aRadialMeshPower)
{
    fRadialMeshPower = aRadialMeshPower;
    return;
}
void KGAnnulusSurface::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGAnnulusSurface::Z() const
{
    return fZ;
}
const double& KGAnnulusSurface::R1() const
{
    return fR1;
}
const double& KGAnnulusSurface::R2() const
{
    return fR2;
}
const unsigned int& KGAnnulusSurface::RadialMeshCount() const
{
    return fRadialMeshCount;
}
const double& KGAnnulusSurface::RadialMeshPower() const
{
    return fRadialMeshPower;
}
const unsigned int& KGAnnulusSurface::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGAnnulusSurface::AreaInitialize() const
{
    if (fR1 < fR2) {
        fPath->X1(fZ);
        fPath->Y1(fR1);
        fPath->X2(fZ);
        fPath->Y2(fR2);
    }
    else {
        fPath->X1(fZ);
        fPath->Y1(fR2);
        fPath->X2(fZ);
        fPath->Y2(fR1);
    }

    fPath->MeshCount(fRadialMeshCount);
    fPath->MeshPower(fRadialMeshPower);
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedLineSegmentSurface::AreaInitialize();
    return;
}
void KGAnnulusSurface::AreaAccept(KGVisitor* aVisitor)
{
    auto* tAnnulusSurfaceVisitor = dynamic_cast<KGAnnulusSurface::Visitor*>(aVisitor);
    if (tAnnulusSurfaceVisitor != nullptr) {
        tAnnulusSurfaceVisitor->VisitAnnulusSurface(this);
        return;
    }
    KGRotatedLineSegmentSurface::AreaAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
