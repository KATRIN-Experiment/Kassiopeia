#ifndef KZHTYPES_H
#define KZHTYPES_H

#include "KSurface.hh"
#include "KTypelist.hh"

namespace KEMField
{

class KConicSection;
class KRing;

using KZHElectrostaticSurfaceTypes_ = KTYPELIST_2(KConicSection, KRing);

class KCurrentLoop;
class KSolenoid;
class KCoil;

using KZHElectromagnetTypes_ =
    KEMField::KTypelist<KCurrentLoop, KEMField::KTypelist<KSolenoid, KEMField::KTypelist<KCoil, KEMField::KNullType>>>;

using KZHElectrostaticSurfaceTypes = NoDuplicates<KZHElectrostaticSurfaceTypes_>::Result;
using KZHElectromagnetTypes = NoDuplicates<KZHElectromagnetTypes_>::Result;
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
    using BaseElement = KElectromagnet;
    using Container = KElectromagnetContainer;
    using Integrator = KElectromagnetIntegrator;
    using Types = KZHElectromagnetTypes;
    using Visitor = KElectromagnetVisitor;
};

template<> struct KZonalHarmonicTrait<KElectrostaticBasis>
{
    static std::string Name()
    {
        return "KZHElectrostaticSurface";
    }
    using BaseElement = KSurfacePrimitive;
    using Container = KSurfaceContainer;
    using Integrator = KElectrostaticBoundaryIntegrator;
    using Types = KZHElectrostaticSurfaceTypes;
    using Visitor = KShapeVisitor;
};
}  // namespace KEMField

#endif /* KZHTYPES */
