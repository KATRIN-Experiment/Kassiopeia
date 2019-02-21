/*
 * KElectrostaticBoundaryFieldBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */
#include "KEMToolboxBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin {

template< >
KElectrostaticBoundaryFieldBuilder::~KComplexElement()
{
}

STATICINT sKEMToolBoxBuilder =
        KEMToolboxBuilder::ComplexElement< KGElectrostaticBoundaryField >( "electrostatic_field" );

STATICINT sKElectrostaticBoundaryFieldBuilder =
		KElectrostaticBoundaryFieldBuilder::Attribute<std::string>( "name" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "directory" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "file" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "system" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "surfaces" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "spaces" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< string >( "symmetry" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< unsigned int >( "hash_masked_bits" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< double >( "hash_threshold" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< double >( "minimum_element_area" ) +
		KElectrostaticBoundaryFieldBuilder::Attribute< double >( "maximum_element_aspect_ratio" );
} //katrin


