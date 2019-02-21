/*
 * KElectricFastMultipoleFieldSolverBuilder.hh
 *
 *  Created on: 24 Jul 2015
 *      Author: wolfgang
 */

#ifndef KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_
#define KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectricFastMultipoleFieldSolver.hh"

namespace katrin {

typedef KComplexElement<KEMField::KElectricFastMultipoleFieldSolver>
KElectricFastMultipoleFieldSolverBuilder;

template< >
inline bool KElectricFastMultipoleFieldSolverBuilder::AddAttribute( KContainer* aContainer )
{
	if( aContainer->GetName() == "top_level_divisions" )
	{
		fObject->GetParameters()->top_level_divisions = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "tree_level_divisions" )
	{
		fObject->GetParameters()->divisions = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "expansion_degree" )
	{
		fObject->GetParameters()->degree = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "neighbor_order" )
	{
		fObject->GetParameters()->zeromask = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "maximum_tree_depth" )
	{
		fObject->GetParameters()->maximum_tree_depth = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "region_expansion_factor" )
	{
		fObject->GetParameters()->region_expansion_factor = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "use_region_size_estimation" )
	{
		fObject->GetParameters()->use_region_estimation = aContainer->AsReference<bool>();
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_x" )
	{
		fObject->GetParameters()->world_center_x = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_y" )
	{
		fObject->GetParameters()->world_center_y = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_z" )
	{
		fObject->GetParameters()->world_center_z = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "world_cube_length" )
	{
		fObject->GetParameters()->world_length = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "use_caching" )
	{
		fObject->GetParameters()->use_caching = aContainer->AsReference<bool>();
		return true;
	}
	if( aContainer->GetName() == "verbosity" )
	{
		fObject->GetParameters()->verbosity = aContainer->AsReference<unsigned int>();
		return true;
	}
	if( aContainer->GetName() == "insertion_ratio" )
	{
		fObject->GetParameters()->insertion_ratio = aContainer->AsReference<double>();
		return true;
	}
	if( aContainer->GetName() == "use_opencl" )
	{
        aContainer->CopyTo(fObject, &KEMField::KElectricFastMultipoleFieldSolver::UseOpenCL);
		return true;
	}
	return false;
}


} /* namespace katrin */
#endif /* KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_ */
