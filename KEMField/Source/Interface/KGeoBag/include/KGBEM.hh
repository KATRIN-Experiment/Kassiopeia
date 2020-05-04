#ifndef KGBEM_DEF
#define KGBEM_DEF

#include "KSurfaceContainer.hh"

using KEMField::KBasisTypes;
using KEMField::KBoundaryType;
using KEMField::KElectromagneticBasis;
using KEMField::KElectrostaticBasis;
using KEMField::KMagnetostaticBasis;

using KEMField::KBoundaryTypes;
using KEMField::KCauchyBoundary;
using KEMField::KDirichletBoundary;
using KEMField::KNeumannBoundary;
using KEMField::KRobinBoundary;

using KEMField::KDirection;
using KEMField::KPosition;
using KGeoBag::KThreeVector;

using KEMField::KConicSection;
using KEMField::KLineSegment;
using KEMField::KRectangle;
using KEMField::KRing;
using KEMField::KShapeTypes;
using KEMField::KSymmetryGroup;
using KEMField::KTriangle;

using KEMField::KNullType;
using KEMField::KSurface;
using KEMField::KSurfaceContainer;
using KEMField::KTypelist;

#include "KGCore.hh"

namespace KGeoBag
{
template<class BasisPolicy, class BoundaryPolicy>
class KGBEMData : public BasisPolicy, public KEMField::KBoundaryType<BasisPolicy, BoundaryPolicy>
{
  public:
    KGBEMData() : BasisPolicy(), KBoundaryType<BasisPolicy, BoundaryPolicy>() {}
    KGBEMData(KGSurface*) : BasisPolicy(), KBoundaryType<BasisPolicy, BoundaryPolicy>() {}
    KGBEMData(KGSpace*) : BasisPolicy(), KBoundaryType<BasisPolicy, BoundaryPolicy>() {}

    KGBEMData(KGSurface*, const KGBEMData& aCopy) :
        BasisPolicy(aCopy),
        KBoundaryType<BasisPolicy, BoundaryPolicy>(aCopy)
    {}
    KGBEMData(KGSpace*, const KGBEMData& aCopy) : BasisPolicy(aCopy), KBoundaryType<BasisPolicy, BoundaryPolicy>(aCopy)
    {}

    ~KGBEMData() override {}

  public:
    typedef BasisPolicy Basis;
    typedef KEMField::KBoundaryType<BasisPolicy, BoundaryPolicy> Boundary;

    Basis* GetBasis()
    {
        return this;
    }
    Boundary* GetBoundary()
    {
        return this;
    }
};

template<class BasisPolicy, class BoundaryPolicy> class KGBEM
{
  public:
    typedef KGBEMData<BasisPolicy, BoundaryPolicy> Surface;
    typedef KGBEMData<BasisPolicy, BoundaryPolicy> Space;
};

typedef KGBEM<KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary> KGElectrostaticDirichlet;
typedef KGBEM<KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary> KGElectrostaticNeumann;
typedef KGBEM<KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary> KGMagnetostaticDirichlet;
typedef KGBEM<KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary> KGMagnetostaticNeumann;

}  // namespace KGeoBag

#endif /* KGBEMDATA_DEF */
