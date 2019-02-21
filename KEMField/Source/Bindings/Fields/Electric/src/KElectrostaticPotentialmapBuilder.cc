/*
 * KElectricPotentialmapBuilder.cc
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#include "KElectrostaticPotentialmapBuilder.hh"

#include "KElectricQuadrupoleFieldBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KElectrostaticConstantFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
namespace katrin {

template< >
KElectrostaticPotentialmapBuilder::~KComplexElement()
{
}

STATICINT sKElectrostaticPotentialmapStructure =
        KElectrostaticPotentialmapBuilder::Attribute< string >( "name" ) +
        KElectrostaticPotentialmapBuilder::Attribute< string >( "directory" ) +
        KElectrostaticPotentialmapBuilder::Attribute< string >( "file" ) +
        KElectrostaticPotentialmapBuilder::Attribute< string >( "interpolation" );

STATICINT sKElectrostaticPotentialmap =
        KEMToolboxBuilder::ComplexElement< KElectrostaticPotentialmap >( "electric_potentialmap" );

////////////////////////////////////////////////////////////////////

template< >
KElectrostaticPotentialmapCalculatorBuilder::~KComplexElement()
{
}

STATICINT sKElectrostaticPotentialmapCalculatorStructure =
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< string >( "name" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< string >( "directory" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< string >( "file" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< KEMStreamableThreeVector >( "center" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< KEMStreamableThreeVector >( "length" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_x" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_y" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_z" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< double >( "spacing" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< string >( "spaces" ) +
        KElectrostaticPotentialmapCalculatorBuilder::Attribute< string >( "field") +
        // support of deprecated old xml:
        KElectrostaticPotentialmapCalculatorBuilder::
        ComplexElement< KElectrostaticConstantField >( "field_electric_constant" ) +
        KElectrostaticPotentialmapCalculatorBuilder::
        ComplexElement< KElectricQuadrupoleField >( "field_electric_quadrupole" ) +
        KElectrostaticPotentialmapCalculatorBuilder::
        ComplexElement< KGElectrostaticBoundaryField >( "field_electrostatic" );

STATICINT sKElectrostaticPotentialmapCalculator =
        KEMToolboxBuilder::ComplexElement< KElectrostaticPotentialmapCalculator >( "electric_potentialmap_calculator" );


} /* namespace katrin */
