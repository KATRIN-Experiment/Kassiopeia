#include "KGCutTorusSurface.hh"

namespace KGeoBag
{

KGCutTorusSurface::Visitor::Visitor() = default;
KGCutTorusSurface::Visitor::~Visitor() = default;

KGCutTorusSurface::KGCutTorusSurface() :
    fZ1(0.),
    fR1(0.),
    fZ2(0.),
    fR2(0.),
    fRadius(0.),
    fRight(true),
    fShort(true),
    fToroidalMeshCount(64),
    fAxialMeshCount(64)
{}
KGCutTorusSurface::~KGCutTorusSurface() = default;

void KGCutTorusSurface::Z1(const double& aZ1)
{
    fZ1 = aZ1;
    return;
}
void KGCutTorusSurface::R1(const double& anR1)
{
    fR1 = anR1;
    return;
}
void KGCutTorusSurface::Z2(const double& aZ2)
{
    fZ2 = aZ2;
    return;
}
void KGCutTorusSurface::R2(const double& anR2)
{
    fR2 = anR2;
    return;
}
void KGCutTorusSurface::Radius(const double& aRadius)
{
    fRadius = aRadius;
    return;
}
void KGCutTorusSurface::Right(const bool& aRight)
{
    fRight = aRight;
    return;
}
void KGCutTorusSurface::Short(const bool& aShort)
{
    fShort = aShort;
    return;
}
void KGCutTorusSurface::ToroidalMeshCount(const unsigned int& aToroidalMeshCount)
{
    fToroidalMeshCount = aToroidalMeshCount;
    return;
}
void KGCutTorusSurface::AxialMeshCount(const unsigned int& anAxialMeshCount)
{
    fAxialMeshCount = anAxialMeshCount;
    return;
}

const double& KGCutTorusSurface::Z1() const
{
    return fZ1;
}
const double& KGCutTorusSurface::R1() const
{
    return fR1;
}
const double& KGCutTorusSurface::Z2() const
{
    return fZ2;
}
const double& KGCutTorusSurface::R2() const
{
    return fR2;
}
const double& KGCutTorusSurface::Radius() const
{
    return fRadius;
}
const bool& KGCutTorusSurface::Right() const
{
    return fRight;
}
const bool& KGCutTorusSurface::Short() const
{
    return fShort;
}
const unsigned int& KGCutTorusSurface::ToroidalMeshCount() const
{
    return fToroidalMeshCount;
}
const unsigned int& KGCutTorusSurface::AxialMeshCount() const
{
    return fAxialMeshCount;
}

void KGCutTorusSurface::AreaInitialize() const
{
    if (fZ1 < fZ2) {
        fPath->X1(fZ2);
        fPath->Y2(fR2);
        fPath->X2(fZ1);
        fPath->Y2(fR1);
        fPath->Radius(fRadius);
        fPath->Right(!fRight);
        fPath->Short(fShort);
    }
    else {
        fPath->X1(fZ1);
        fPath->Y2(fR1);
        fPath->X2(fZ2);
        fPath->Y2(fR2);
        fPath->Radius(fRadius);
        fPath->Right(fRight);
        fPath->Short(fShort);
    }

    fPath->MeshCount(fToroidalMeshCount);
    fRotatedMeshCount = fAxialMeshCount;

    KGRotatedArcSegmentSurface::AreaInitialize();
    return;
}
void KGCutTorusSurface::AreaAccept(KGVisitor* aVisitor)
{
    auto* tCutTorusSurfaceVisitor = dynamic_cast<KGCutTorusSurface::Visitor*>(aVisitor);
    if (tCutTorusSurfaceVisitor != nullptr) {
        tCutTorusSurfaceVisitor->VisitCutTorusSurface(this);
        return;
    }
    KGRotatedArcSegmentSurface::AreaAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
