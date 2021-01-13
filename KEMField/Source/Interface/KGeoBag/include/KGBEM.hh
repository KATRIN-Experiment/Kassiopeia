#ifndef KGBEM_DEF
#define KGBEM_DEF

#include "KSurfaceContainer.hh"

using KEMField::KBasisTypes;
using KEMField::KBoundaryType;
using KEMField::KElectrostaticBasis;
using KEMField::KMagnetostaticBasis;

using KEMField::KBoundaryTypes;
using KEMField::KDirichletBoundary;
using KEMField::KNeumannBoundary;

using KEMField::KDirection;
using KEMField::KPosition;

using KEMField::KShapeTypes;


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

    ~KGBEMData() override = default;

  public:
    using Basis = BasisPolicy;
    using Boundary = KEMField::KBoundaryType<BasisPolicy, BoundaryPolicy>;

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
    using Surface = KGBEMData<BasisPolicy, BoundaryPolicy>;
    using Space = KGBEMData<BasisPolicy, BoundaryPolicy>;
};

using KGElectrostaticDirichlet = KGBEM<KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary>;
using KGElectrostaticNeumann = KGBEM<KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary>;
using KGMagnetostaticDirichlet = KGBEM<KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary>;
using KGMagnetostaticNeumann = KGBEM<KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary>;

}  // namespace KGeoBag

#endif /* KGBEMDATA_DEF */
