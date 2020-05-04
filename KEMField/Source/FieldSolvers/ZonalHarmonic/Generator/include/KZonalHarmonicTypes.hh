#ifndef KZHTYPES_H
#define KZHTYPES_H

#include "KSurface.hh"
#include "KTypelist.hh"

namespace KEMField
{

class KConicSection;
class KRing;

typedef KTYPELIST_2(KConicSection, KRing) KZHElectrostaticSurfaceTypes_;

class KCurrentLoop;
class KSolenoid;
class KCoil;

typedef KTYPELIST_3(KCurrentLoop, KSolenoid, KCoil) KZHElectromagnetTypes_;

typedef NoDuplicates<KZHElectrostaticSurfaceTypes_>::Result KZHElectrostaticSurfaceTypes;
typedef NoDuplicates<KZHElectromagnetTypes_>::Result KZHElectromagnetTypes;
}  // namespace KEMField

#include "KElectromagnetContainer.hh"
#include "KElectromagnetIntegrator.hh"
#include "KElectrostaticBoundaryIntegrator.hh"
#include "KSurfaceContainer.hh"
#include "KZHCoefficientGeneratorCoil.hh"
#include "KZHCoefficientGeneratorConicSection.hh"
#include "KZHCoefficientGeneratorCurrentLoop.hh"
#include "KZHCoefficientGeneratorRing.hh"
#include "KZHCoefficientGeneratorSolenoid.hh"

namespace KEMField
{
template<typename Basis> struct KZonalHarmonicTrait;

template<> struct KZonalHarmonicTrait<KMagnetostaticBasis>
{
    static std::string Name()
    {
        return "KZHElectromagnet";
    }
    typedef KElectromagnet BaseElement;
    typedef KElectromagnetContainer Container;
    typedef KElectromagnetIntegrator Integrator;
    typedef KZHElectromagnetTypes Types;
    typedef KElectromagnetVisitor Visitor;
};

template<> struct KZonalHarmonicTrait<KElectrostaticBasis>
{
    static std::string Name()
    {
        return "KZHElectrostaticSurface";
    }
    typedef KSurfacePrimitive BaseElement;
    typedef KSurfaceContainer Container;
    typedef KElectrostaticBoundaryIntegrator Integrator;
    typedef KZHElectrostaticSurfaceTypes Types;
    typedef KShapeVisitor Visitor;
};
}  // namespace KEMField

#endif /* KZHTYPES */
