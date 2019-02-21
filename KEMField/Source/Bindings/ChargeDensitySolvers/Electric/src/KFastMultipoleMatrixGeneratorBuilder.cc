/*
 * KFastMultipoleMatrixGeneratorBuilder.cc
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#include "KFastMultipoleMatrixGeneratorBuilder.hh"

#include "KKrylovChargeDensitySolverBuilder.hh"
#include "KKrylovPreconditionerGeneratorBuilder.hh"

using namespace KEMField;

namespace katrin {

template< >
KFastMultipoleMatrixGeneratorBuilder::~KComplexElement() {
}

STATICINT sKFastMultipoleMatrixGeneratorStructure =
    KFastMultipoleMatrixGeneratorBuilder::
            Attribute< std::string >( "integrator" ) +
    KFastMultipoleMatrixGeneratorBuilder::
            Attribute< std::string >( "strategy" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< unsigned int >( "top_level_divisions" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< unsigned int >( "tree_level_divisions" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< unsigned int >( "expansion_degree" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< unsigned int >( "neighbor_order" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< unsigned int >( "maximum_tree_depth" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< double >( "region_expansion_factor" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< bool >( "use_region_size_estimation" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< KEMStreamableThreeVector >( "world_cube_center" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< double >( "world_cube_length" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< bool >( "use_caching" ) +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute<unsigned int>("verbosity") +
	KFastMultipoleMatrixGeneratorBuilder::
			Attribute< double >( "insertion_ratio" ) +
	KFastMultipoleMatrixGeneratorBuilder::
	        Attribute< unsigned int >("allowed_number") +
	KFastMultipoleMatrixGeneratorBuilder::
	        Attribute< double >("allowed_fraction") +
	KFastMultipoleMatrixGeneratorBuilder::
	        Attribute< double >( "bias_degree" );

template<class Builder>
int AddMatrixAsElement(){
	return Builder::template
			ComplexElement< KFastMultipoleMatrixGenerator>("fast_multipole_matrix") +
	Builder::template
			ComplexElement< KFastMultipoleMatrixGenerator>("fftm_multiplication");
}

STATICINT sKKrylovChargeDensitySolverStructure =
		AddMatrixAsElement<KKrylovChargeDensitySolverBuilder>();

STATICINT sKKrylvoChargeDensitySolverStructure =
		AddMatrixAsElement< KKrylovPreconditionerGeneratorBuilder>();

} /* namespace katrin */
