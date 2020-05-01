#include "KGAxialMeshRing.hh"

#include "KConst.h"

namespace KGeoBag
{

KGAxialMeshRing::KGAxialMeshRing(const double& aD, const KTwoVector& aP0) : fD(aD), fP0(aP0) {}
KGAxialMeshRing::~KGAxialMeshRing() {}

double KGAxialMeshRing::Area() const
{
    return 2. * std::pow(katrin::KConst::Pi(), 2) * fP0.Y() * fD;
}
double KGAxialMeshRing::Aspect() const
{
    return fD / fP0.Y();
}

}  // namespace KGeoBag
