/*
 * KKrylovPreconditionerGeneratorBuilder.cc
 *
 *  Created on: 19 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovPreconditionerGeneratorBuilder.hh"

#include "KKrylovChargeDensitySolverBuilder.hh"

using namespace KEMField;

namespace katrin {

template <>
KKrylovPreconditionerGeneratorBuilder::~KComplexElement() {
}

STATICINT sKKrylovPreconditionerGeneratorStructure =
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< std::string >( "solver_name" ) +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< double >("tolerance") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< unsigned int >("max_iterations") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< unsigned int >("iterations_between_restarts") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< unsigned int >("intermediate_save_interval") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< bool >("use_display") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< bool >("show_plot") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< bool >("use_timer") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< double >("time_limit_in_seconds") +
	KKrylovPreconditionerGeneratorBuilder::
		Attribute< unsigned int >("time_check_interval");

STATICINT sKKrylovChargeDensitySolverStructure =
		KKrylovChargeDensitySolverBuilder::
			ComplexElement< KKrylovPreconditionerGenerator >( "krylov_preconditioner" ) +
		KKrylovPreconditionerGeneratorBuilder::
			ComplexElement< KKrylovPreconditionerGenerator >( "krylov_preconditioner" );


} /* namespace katrin */
