#ifndef KSURFACETYPES_DEF
#define KSURFACETYPES_DEF

#include <complex>

#include "KTypelist.hh"

namespace KEMField
{
  class KElectrostaticBasis;
  class KMagnetostaticBasis;
  class KElectromagneticBasis;

  // A list of all of the basis types
  typedef KTYPELIST_3( KElectrostaticBasis,
		       KMagnetostaticBasis,
		       KElectromagneticBasis) KBasisTypes_;
}

namespace KEMField
{
  struct KDirichletBoundary;
  struct KNeumannBoundary;
  struct KCauchyBoundary;
  struct KRobinBoundary;
  struct KIsolatedBoundary;

  // A list of all of the boundary types
  typedef KTYPELIST_5( KDirichletBoundary,
		       KNeumannBoundary,
		       KCauchyBoundary,
		       KRobinBoundary,
		       KIsolatedBoundary) KBoundaryTypes_;
}

namespace KEMField
{
  class KTriangle;
  class KRectangle;
  class KLineSegment;
  class KConicSection;
  class KRing;
  template <class ShapePolicy>
  class KSymmetryGroup;

  typedef KSymmetryGroup<KTriangle> KTriangleGroup;
  typedef KSymmetryGroup<KRectangle> KRectangleGroup;
  typedef KSymmetryGroup<KLineSegment> KLineSegmentGroup;
  typedef KSymmetryGroup<KConicSection> KConicSectionGroup;
  typedef KSymmetryGroup<KRing> KRingGroup;

  // A list of all of the shape types
  typedef KTYPELIST_10( KTriangle,
			KRectangle,
			KLineSegment,
			KConicSection,
			KRing,
			KRectangleGroup,
			KLineSegmentGroup,
			KTriangleGroup,
			KConicSectionGroup,
			KRingGroup) KShapeTypes_;
}

namespace KEMField
{
/**
* KBasisTypes is a typelist of all available basis types.
*/
  typedef NoDuplicates<KBasisTypes_>::Result KBasisTypes;
/**
* KBoundaryTypes is a typelist of all available boundary types.
*/
  typedef NoDuplicates<KBoundaryTypes_>::Result KBoundaryTypes;
/**
* KShapeTypes is a typelist of all available shape types.
*/
  typedef NoDuplicates<KShapeTypes_>::Result KShapeTypes;
}

#include "KSurfaceVisitors.hh"

#include "KElectrostaticBasis.hh"
#include "KMagnetostaticBasis.hh"
#include "KElectromagneticBasis.hh"

#include "KBoundary.hh"

#include "KTriangle.hh"
#include "KRectangle.hh"
#include "KLineSegment.hh"
#include "KConicSection.hh"
#include "KRing.hh"
#include "KSymmetryGroup.hh"

#endif /* KSURFACETYPES_DEF */
