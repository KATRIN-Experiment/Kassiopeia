#include "KGAxialMeshLoop.hh"

#include "KConst.h"

namespace KGeoBag
{

KGAxialMeshLoop::KGAxialMeshLoop(const KTwoVector& aP0, const KTwoVector& aP1) : fP0(aP0), fP1(aP1) {}
KGAxialMeshLoop::~KGAxialMeshLoop() = default;

double KGAxialMeshLoop::Area() const
{
    return katrin::KConst::Pi() * (fP0 - fP1).Magnitude() * (fP0 + fP1).Y();
}
double KGAxialMeshLoop::Aspect() const
{
    return 2. * (fP0 - fP1).Magnitude() / (fP0 + fP1).Y();
}

}  // namespace KGeoBag
