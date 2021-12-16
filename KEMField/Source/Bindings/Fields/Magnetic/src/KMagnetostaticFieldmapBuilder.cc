/*
 * KMagnetostaticFieldmapBuilder.cc
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#include "KMagnetostaticFieldmapBuilder.hh"

#include "KEMToolboxBuilder.hh"
#include "KMagneticDipoleFieldBuilder.hh"
#include "KMagnetostaticConstantFieldBuilder.hh"
#include "KStaticElectromagnetFieldBuilder.hh"

using namespace KEMField;
namespace katrin
{

template<> KMagnetostaticFieldmapBuilder::~KComplexElement() = default;

STATICINT sKMagnetostaticFieldmapStructure = KMagnetostaticFieldmapBuilder::Attribute<std::string>("name") +
                                             KMagnetostaticFieldmapBuilder::Attribute<std::string>("directory") +
                                             KMagnetostaticFieldmapBuilder::Attribute<std::string>("file") +
                                             KMagnetostaticFieldmapBuilder::Attribute<std::string>("interpolation") +
                                             KMagnetostaticFieldmapBuilder::Attribute<bool>("magnetic_gradient_numerical");

STATICINT sKMagnetostaticFieldmap = KEMToolboxBuilder::ComplexElement<KMagnetostaticFieldmap>("magnetic_fieldmap");

////////////////////////////////////////////////////////////////////

template<> KMagnetostaticFieldmapCalculatorBuilder::~KComplexElement() = default;

STATICINT sKMagnetostaticFieldmapCalculatorStructure =
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<std::string>("name") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<std::string>("directory") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<std::string>("file") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<bool>("force_update") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<bool>("compute_gradient") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("center") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("length") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<bool>("mirror_x") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<bool>("mirror_y") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<bool>("mirror_z") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<double>("spacing") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<double>("time") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<std::string>("spaces") +
    KMagnetostaticFieldmapCalculatorBuilder::Attribute<std::string>("field") +
    // support of deprecated old xml:
    KMagnetostaticFieldmapCalculatorBuilder::ComplexElement<KMagnetostaticConstantField>("field_magnetic_constant") +
    KMagnetostaticFieldmapCalculatorBuilder::ComplexElement<KMagneticDipoleFieldBuilder>("field_magnetic_dipole") +
    KMagnetostaticFieldmapCalculatorBuilder::ComplexElement<KStaticElectromagnetField>("field_electromagnet");

STATICINT sKMagnetostaticFieldmapCalculator =
    KEMToolboxBuilder::ComplexElement<KMagnetostaticFieldmapCalculator>("magnetic_fieldmap_calculator");


} /* namespace katrin */
