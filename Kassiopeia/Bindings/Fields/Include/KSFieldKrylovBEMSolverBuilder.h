/*
 * KSFieldFastMultipoleBEMSolverBuilder.h
 *
 *  Created on: 27.04.2015
 *      Author: gosda
 */

#ifndef KSFIELDFASTMULTIPOLEBEMSOLVERBUILDER_H_
#define KSFIELDFASTMULTIPOLEBEMSOLVERBUILDER_H_

#include "KComplexElement.hh"
#include "KSFieldKrylovBEMSolver.h"
#include "KSFieldFMElectrostaticParametersBuilder.h"

using namespace KEMField;
namespace katrin
{

typedef KComplexElement< KrylovBEMSolver > KSKrylovBEMSolverBuilder;

template< >
inline bool KSKrylovBEMSolverBuilder::
	AddAttribute(KContainer* aContainer)
{
	if ( aContainer->GetName() == "solver_name"){
                fObject->GetSolverConfig()->SetSolverName( aContainer->AsReference<std::string>() );
		return true;
	}
	if (aContainer->GetName() == "preconditioner"){
                fObject->GetSolverConfig()->SetPreconditionerName( aContainer->AsReference<std::string>() );
		return true;
	}
	if (aContainer->GetName() == "tolerance"){
                fObject->GetSolverConfig()->SetSolverTolerance( aContainer->AsReference<double>() );
		return true;
	}
	if (aContainer->GetName() == "max_iterations"){
                fObject->GetSolverConfig()->SetMaxSolverIterations(aContainer->AsReference<unsigned int>());
		return true;
	}
	if (aContainer->GetName() == "iterations_between_restarts"){
                fObject->GetSolverConfig()->SetIterationsBetweenRestart(aContainer->AsReference<unsigned int>());
		return true;
	}
	if(aContainer->GetName() == "preconditioner_tolerance"){
                fObject->GetSolverConfig()->SetPreconditionerTolerance(aContainer->AsReference<double>());
		return true;
	}
	if(aContainer->GetName() == "max_preconditioner_iterations"){
                fObject->GetSolverConfig()->SetMaxPreconditionerIterations(aContainer->AsReference<unsigned int>());
		return true;
	}
	if(aContainer->GetName() == "preconditioner_degree"){
                fObject->GetSolverConfig()->SetPreconditionerDegree(aContainer->AsReference<unsigned int>());
		return true;
	}
	if(aContainer->GetName() == "intermediate_save_interval"){
                fObject->GetSolverConfig()->SetUseCheckpoints(aContainer->AsReference<unsigned int>());
                fObject->GetSolverConfig()->SetCheckpointFrequency(aContainer->AsReference<unsigned int>());
		return true;
	}
	if(aContainer->GetName() == "use_display"){
                fObject->GetSolverConfig()->SetUseDisplay(aContainer->AsReference<bool>());
		return true;
	}
	if(aContainer->GetName() == "show_plot"){
                fObject->GetSolverConfig()->SetUsePlot(aContainer->AsReference<bool>());
		return true;
	}
	if(aContainer->GetName() == "use_timer"){
                fObject->GetSolverConfig()->SetUseTimer(aContainer->AsReference<bool>());
		return true;
	}
	if(aContainer->GetName() == "time_limit_in_seconds"){
                fObject->GetSolverConfig()->SetTimeLimitSeconds(aContainer->AsReference<double>());
		return true;
	}
	if(aContainer->GetName() == "time_check_interval"){
                fObject->GetSolverConfig()->SetTimeCheckFrequency(aContainer->AsReference<unsigned int>());
		return true;
	}
	return false;
}

template< >
inline bool KSKrylovBEMSolverBuilder::
		AddElement(KContainer * anElement)
{
	if(anElement->GetName() == "fftm_multiplication"){
		anElement->ReleaseTo(fObject->GetSolverConfig(),
				&KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::
				SetFFTMParams);
		return true;
	}
	if(anElement->GetName() == "preconditioner_electrostatic_parameters"){
		anElement->ReleaseTo(fObject->GetSolverConfig(),
				&KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::
				SetPreconditionerFFTMParams);
		return true;
	}
	return false;
}

}

#endif /* KSFIELDFASTMULTIPOLEBEMSOLVERBUILDER_H_ */
