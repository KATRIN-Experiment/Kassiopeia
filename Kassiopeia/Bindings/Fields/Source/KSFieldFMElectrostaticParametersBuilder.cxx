/*
 * KSFieldFastMultipoleBEMSolverBuilder.cxx
 *
 *  Created on: 20.04.2015
 *      Author: gosda
 */
#include "KSFieldFMElectrostaticParametersBuilder.h"
#include "KSRootBuilder.h"

using namespace KEMField;
namespace katrin
{

template< >
KSFieldFMElectrostaticParametersBuilder::~KComplexElement()
{
}

STATICINT sKSFieldFMElectrostaticParametersStructure =
	KSFieldFMElectrostaticParametersBuilder::Attribute< string >( "strategy" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< unsigned int >( "top_level_divisions" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< unsigned int >( "tree_level_divisions" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< unsigned int >( "expansion_degree" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< unsigned int >( "neighbor_order" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< unsigned int >( "maximum_tree_depth" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "region_expansion_factor" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< bool >( "use_region_size_estimation" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "world_cube_center_x" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "world_cube_center_y" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "world_cube_center_z" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "world_cube_length" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute< bool >( "use_caching" ) +
	KSFieldFMElectrostaticParametersBuilder::Attribute<unsigned int>("verbosity") +
	KSFieldFMElectrostaticParametersBuilder::Attribute<unsigned int>("allowed_number") +
	KSFieldFMElectrostaticParametersBuilder::Attribute<double>("allowed_fraction") +
	KSFieldFMElectrostaticParametersBuilder::Attribute<unsigned int>("bias_degree") +
	KSFieldFMElectrostaticParametersBuilder::Attribute< double >( "insertion_ratio" );

} //katrin
