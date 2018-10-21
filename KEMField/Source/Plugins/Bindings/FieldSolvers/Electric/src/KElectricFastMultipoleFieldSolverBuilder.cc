/*
 * KElectricFastMultipoleFieldSolverBuilder.cc
 *
 *  Created on: 24 Jul 2015
 *      Author: wolfgang
 */

#include "KElectricFastMultipoleFieldSolverBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin {

template< >
KElectricFastMultipoleFieldSolverBuilder::~KComplexElement()
{
}

STATICINT sKElectricFastMultipoleFieldSolverStructure =
		KElectricFastMultipoleFieldSolverBuilder::Attribute< unsigned int >( "top_level_divisions" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< unsigned int >( "tree_level_divisions" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< unsigned int >( "expansion_degree" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< unsigned int >( "neighbor_order" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< unsigned int >( "maximum_tree_depth" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "region_expansion_factor" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< bool >( "use_region_size_estimation" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "world_cube_center_x" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "world_cube_center_y" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "world_cube_center_z" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "world_cube_length" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< bool >( "use_caching" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("verbosity") +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< double >( "insertion_ratio" ) +
		KElectricFastMultipoleFieldSolverBuilder::Attribute< bool >( "use_opencl" );

STATICINT sKElectrostaticBoundaryField =
		KElectrostaticBoundaryFieldBuilder::ComplexElement<KElectricFastMultipoleFieldSolver>(
				"fast_multipole_field_solver" );

} /* namespace katrin */
