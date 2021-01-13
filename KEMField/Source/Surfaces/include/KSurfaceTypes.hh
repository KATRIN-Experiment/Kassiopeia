#ifndef KSURFACETYPES_DEF
#define KSURFACETYPES_DEF

#include "KTypelist.hh"

#include <complex>

namespace KEMField
{
class KElectrostaticBasis;
class KMagnetostaticBasis;
class KElectromagneticBasis;

// A list of all of the basis types
typedef KTYPELIST_3(KElectrostaticBasis, KMagnetostaticBasis, KElectromagneticBasis) KBasisTypes_;
}  // namespace KEMField

namespace KEMField
{
struct KDirichletBoundary;
struct KNeumannBoundary;
struct KCauchyBoundary;
struct KRobinBoundary;
struct KIsolatedBoundary;

// A list of all of the boundary types
using KBoundaryTypes_ = KEMField::KTypelist<
    KDirichletBoundary,
    KEMField::KTypelist<
        KNeumannBoundary,
        KEMField::KTypelist<
            KCauchyBoundary,
            KEMField::KTypelist<KRobinBoundary, KEMField::KTypelist<KIsolatedBoundary, KEMField::KNullType>>>>>;
}  // namespace KEMField

namespace KEMField
{
class KTriangle;
class KRectangle;
class KLineSegment;
class KConicSection;
class KRing;
template<class ShapePolicy> class KSymmetryGroup;

using KTriangleGroup = KSymmetryGroup<KTriangle>;
using KRectangleGroup = KSymmetryGroup<KRectangle>;
using KLineSegmentGroup = KSymmetryGroup<KLineSegment>;
using KConicSectionGroup = KSymmetryGroup<KConicSection>;
using KRingGroup = KSymmetryGroup<KRing>;

// A list of all of the shape types
using KShapeTypes_ = KEMField::KTypelist<
    KTriangle,
    KEMField::KTypelist<
        KRectangle,
        KEMField::KTypelist<
            KLineSegment,
            KEMField::KTypelist<
                KConicSection,
                KEMField::KTypelist<
                    KRing,
                    KEMField::KTypelist<
                        KRectangleGroup,
                        KEMField::KTypelist<
                            KLineSegmentGroup,
                            KEMField::KTypelist<
                                KTriangleGroup,
                                KEMField::KTypelist<KConicSectionGroup,
                                                    KEMField::KTypelist<KRingGroup, KEMField::KNullType>>>>>>>>>>;
}  // namespace KEMField

namespace KEMField
{
/**
* KBasisTypes is a typelist of all available basis types.
*/
using KBasisTypes = NoDuplicates<KBasisTypes_>::Result;
/**
* KBoundaryTypes is a typelist of all available boundary types.
*/
using KBoundaryTypes = NoDuplicates<KBoundaryTypes_>::Result;
/**
* KShapeTypes is a typelist of all available shape types.
*/
using KShapeTypes = NoDuplicates<KShapeTypes_>::Result;
}  // namespace KEMField

#include "../../../Surfaces/include/KBoundary.hh"
#include "../../../Surfaces/include/KConicSection.hh"
#include "../../../Surfaces/include/KElectromagneticBasis.hh"
#include "../../../Surfaces/include/KElectrostaticBasis.hh"
#include "../../../Surfaces/include/KLineSegment.hh"
#include "../../../Surfaces/include/KMagnetostaticBasis.hh"
#include "../../../Surfaces/include/KRectangle.hh"
#include "../../../Surfaces/include/KRing.hh"
#include "../../../Surfaces/include/KSurfaceVisitors.hh"
#include "../../../Surfaces/include/KSymmetryGroup.hh"
#include "../../../Surfaces/include/KTriangle.hh"

#endif /* KSURFACETYPES_DEF */
