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
    //! the name of the field
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("name") +
    //! the directory of the potential map image data file
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("directory") +
    //! the name of the potential map image data file
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("file") +
    //! interpolation method:
    //!
    //! - ``none`` or ``nearest``: just take the value of the nearest point of the map
    //! - ``linear``: linear interpolation from surrounding points
    //! - ``cubic``: cubic interpolation from surrounding points
    KElectrostaticPotentialmapBuilder::Attribute<std::string>("interpolation");

STATICINT sKElectrostaticPotentialmap =
    KEMToolboxBuilder::ComplexElement<KElectrostaticPotentialmap>("electric_potentialmap");

////////////////////////////////////////////////////////////////////

template<> KElectrostaticPotentialmapCalculatorBuilder::~KComplexElement() = default;

STATICINT sKElectrostaticPotentialmapCalculatorStructure =
    //! name of the calculator
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("name") +
    //! the directory of the data file.
    //! Defaults to the KEMField scratch directory.
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("directory") +
    //! the name of the image data file containing the map.
    //! Use .vti as extension.
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("file") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("force_update") +
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("compute_field") +
    //! The center of the cubic mapping region
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("center") +
    //! The length of the cube where the map is created
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<KEMStreamableThreeVector>("length") +
    //! boolean option to reduce the map size for mirror symmetric problems
    //! along the x axis
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_x") +
    //! boolean option to reduce the map size for mirror symmetric problems
    //! along the y axis
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_y") +
    //! boolean option to reduce the map size for mirror symmetric problems
    //! along the z axis. These mirror options are combinable.
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<bool>("mirror_z") +
    //! Distance between the evaluation points for the map.
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<double>("spacing") +
    //! Define spaces to confine the map to points within the intersection of
    //! these spaces.
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("spaces") +
    //! the name of the field to map
    KElectrostaticPotentialmapCalculatorBuilder::Attribute<std::string>("field") +
    // support of deprecated old xml:
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KElectrostaticConstantField>(
        "field_electric_constant") +
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KElectricQuadrupoleField>("field_electric_quadrupole") +
    KElectrostaticPotentialmapCalculatorBuilder::ComplexElement<KGElectrostaticBoundaryField>("field_electrostatic");

STATICINT sKElectrostaticPotentialmapCalculator =
    KEMToolboxBuilder::ComplexElement<KElectrostaticPotentialmapCalculator>("electric_potentialmap_calculator");


} /* namespace katrin */
