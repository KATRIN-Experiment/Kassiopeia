/*
 * KSFieldFastMultipoleBEMSolverBuilder.cxx
 *
 *  Created on: 27.04.2015
 *      Author: gosda
 */

#include "KSFieldKrylovBEMSolverBuilder.h"

using namespace KEMField;
namespace katrin {

template< >
KSKrylovBEMSolverBuilder::~KComplexElement()
{
}

STATICINT sKSKrylovBEMSolverStructure =
	KSKrylovBEMSolverBuilder::
		Attribute< std::string >( "solver_name" ) +
	KSKrylovBEMSolverBuilder::
		Attribute< std::string >( "preconditioner" ) +
	KSKrylovBEMSolverBuilder::
		Attribute< double >("tolerance") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("max_iterations") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("iterations_between_restarts") +
	KSKrylovBEMSolverBuilder::
		Attribute< double >("preconditioner_tolerance") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("max_preconditioner_iterations") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("preconditioner_degree") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("intermediate_save_interval") +
	KSKrylovBEMSolverBuilder::
		Attribute< bool >("use_display") +
	KSKrylovBEMSolverBuilder::
		Attribute< bool >("show_plot") +
	KSKrylovBEMSolverBuilder::
		Attribute< bool >("use_timer") +
	KSKrylovBEMSolverBuilder::
		Attribute< double >("time_limit_in_seconds") +
	KSKrylovBEMSolverBuilder::
		Attribute< unsigned int >("time_check_interval") +
	KSKrylovBEMSolverBuilder::
		ComplexElement< KFMElectrostaticParameters >("fftm_multiplication") +
	KSKrylovBEMSolverBuilder::
		ComplexElement< KFMElectrostaticParameters >("preconditioner_electrostatic_parameters");

}//katrin
