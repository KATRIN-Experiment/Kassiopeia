/*
 * KIntegratingElectrostaticFieldSolver.hh
 *
 *  Created on: 05.06.2015
 *      Author: gosda
 *
 *      This is just a wrapper class for bindings or other high level uses.
 *      Imported from KSFieldElectrostatic
 */

#ifndef KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_
#define KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif

namespace KEMField {

class KIntegratingElectrostaticFieldSolver :
		public KElectricFieldSolver
{
public:
	KIntegratingElectrostaticFieldSolver();
	virtual ~KIntegratingElectrostaticFieldSolver();

	void SetIntegratorPolicy( KEBIPolicy& policy)
	{
		fIntegratorPolicy = policy;
	}

	void UseOpenCL( bool choice )
	{
		fUseOpenCL = choice;
	}

private:
	void InitializeCore( KSurfaceContainer& container );

	double PotentialCore( const KPosition& P ) const;
	KThreeVector ElectricFieldCore( const KPosition& P ) const;

	KElectrostaticBoundaryIntegrator* fIntegrator;
	KEBIPolicy fIntegratorPolicy;
	KIntegratingFieldSolver< KElectrostaticBoundaryIntegrator >* fIntegratingFieldSolver;

#ifdef KEMFIELD_USE_OPENCL
KOpenCLElectrostaticBoundaryIntegrator* fOCLIntegrator;
KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >* fOCLIntegratingFieldSolver;
#endif

bool fUseOpenCL;
};

} //KEMField

#endif /* KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_ */
