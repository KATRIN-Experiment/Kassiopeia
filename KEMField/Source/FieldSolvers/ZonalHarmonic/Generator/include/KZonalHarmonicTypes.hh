#ifndef KZHTYPES_H
#define KZHTYPES_H

#include "KTypelist.hh"

#include "KSurface.hh"

namespace KEMField
{

  class KConicSection;
  class KRing;

  typedef KTYPELIST_2( KConicSection,
		       KRing ) KZHElectrostaticSurfaceTypes_;

  class KCurrentLoop;
  class KSolenoid;
  class KCoil;

  typedef KTYPELIST_3( KCurrentLoop,
		       KSolenoid,
		       KCoil ) KZHElectromagnetTypes_;

  typedef NoDuplicates<KZHElectrostaticSurfaceTypes_>::Result KZHElectrostaticSurfaceTypes;
  typedef NoDuplicates<KZHElectromagnetTypes_>::Result KZHElectromagnetTypes;
}

#include "KZHCoefficientGeneratorConicSection.hh"
#include "KZHCoefficientGeneratorRing.hh"
#include "KZHCoefficientGeneratorCurrentLoop.hh"
#include "KZHCoefficientGeneratorSolenoid.hh"
#include "KZHCoefficientGeneratorCoil.hh"

#include "KElectromagnetContainer.hh"
#include "KElectromagnetIntegrator.hh"

#include "KSurfaceContainer.hh"
#include "KElectrostaticBoundaryIntegrator.hh"

namespace KEMField
{
  template <typename Basis>
  struct KZonalHarmonicTrait;

  template <>
  struct KZonalHarmonicTrait<KMagnetostaticBasis>
  {
    static std::string Name() { return "KZHElectromagnet"; }
    typedef KElectromagnet BaseElement;
    typedef KElectromagnetContainer Container;
    typedef KElectromagnetIntegrator Integrator;
    typedef KZHElectromagnetTypes Types;
    typedef KElectromagnetVisitor Visitor;
  };

  template <>
  struct KZonalHarmonicTrait<KElectrostaticBasis>
  {
    static std::string Name() { return "KZHElectrostaticSurface"; }
    typedef KSurfacePrimitive BaseElement;
    typedef KSurfaceContainer Container;
    typedef KElectrostaticBoundaryIntegrator Integrator;
    typedef KZHElectrostaticSurfaceTypes Types;
    typedef KShapeVisitor Visitor;
  };
}

#endif /* KZHTYPES */
