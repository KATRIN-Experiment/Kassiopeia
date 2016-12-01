/*
 * KElectricFastMultipoleFieldSolver.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"

#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KElectrostaticBoundaryIntegratorPolicy.hh"

#ifdef KEMFIELD_USE_OPENCL
//#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#endif

namespace KEMField {

class KElectricFastMultipoleFieldSolver : public KElectricFieldSolver{
public:
	KElectricFastMultipoleFieldSolver();
	virtual ~KElectricFastMultipoleFieldSolver();

	void InitializeCore( KSurfaceContainer& container );

	double PotentialCore( const KPosition& P ) const;
	KEMThreeVector ElectricFieldCore( const KPosition& P ) const;

	void SetIntegratorPolicy(const KEBIPolicy& policy) {
		fIntegratorPolicy = policy;
	}

	KFMElectrostaticParameters* GetParameters()
	{
		return &fParameters;
	}

	void UseOpenCL( bool choice );

private:

	KEBIPolicy fIntegratorPolicy;
	KFMElectrostaticParameters fParameters;
	KFMElectrostaticTree* fTree;
	KFMElectrostaticFastMultipoleFieldSolver* fFastMultipoleFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
	KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fFastMultipoleFieldSolverOpenCL;
#endif
	bool fUseOpenCL;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_ */
