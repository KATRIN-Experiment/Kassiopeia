/*
 * KElectricPotentialmapBuilder.cc
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#include "KElectrostaticPotentialmapBuilder.hh"

#include "KEMToolboxBuilder.hh"
#include "KElectricQuadrupoleFieldBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KElectrostaticConstantFieldBuilder.hh"

using namespace KEMField;
namespace katrin
{

template<> KElectrostaticPotentialmapBuilder::~KComplexElement() = default;

STATICINT sKElectrostaticPotentialmapStructure =
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("name") +
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("directory") +
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("file") +
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("interpolation");

STATICINT sKElectrostaticPotentialmap =
    KEMToolboxBuilder::ComplexElement<KElectrostaticPotentialmap>("electric_potentialmap");

////////////////////////////////////////////////////////////////////

template<> KElectrostaticPotentialmapCalculatorBuilder::~KComplexElement() = default;

STATICINT sKElectrostaticPotentialmapCalculatorStructure =
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("name") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("directory") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("file") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("force_update") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("compute_field") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("center") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("length") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_x") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_y") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_z") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<double>("spacing") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("spaces") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("field") +
    // support of deprecated old xml:
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KElectrostaticConstantField>(
        "field_electric_constant") +
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KElectricQuadrupoleField>("field_electric_quadrupole") +
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KGElectrostaticBoundaryField>("field_electrostatic");

STATICINT sKElectrostaticPotentialmapCalculator =
    KEMToolboxBuilder::ComplexElement<KElectrostaticPotentialmapCalculator>("electric_potentialmap_calculator");


} /* namespace katrin */
