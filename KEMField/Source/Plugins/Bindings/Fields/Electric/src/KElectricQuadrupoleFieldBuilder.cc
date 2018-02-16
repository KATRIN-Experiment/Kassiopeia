/*
 * KElectricQuadrupoleFieldBuilder.cc
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#include "KElectricQuadrupoleFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template< >
KElectricQuadrupoleFieldBuilder::~KComplexElement()
{
}

STATICINT sKElectricQuadrupoleFieldStructure =
		KElectricQuadrupoleFieldBuilder::Attribute< string >( "name" ) +
		KElectricQuadrupoleFieldBuilder::Attribute< KEMStreamableThreeVector >( "location" ) +
		KElectricQuadrupoleFieldBuilder::Attribute< double >( "strength" ) +
		KElectricQuadrupoleFieldBuilder::Attribute< double >( "length" ) +
		KElectricQuadrupoleFieldBuilder::Attribute< double >( "radius" );

STATICINT sKToolboxStructure =
        KEMToolboxBuilder::ComplexElement< KElectricQuadrupoleField >( "electric_quadrupole_field");


} /* namespace katrin */
