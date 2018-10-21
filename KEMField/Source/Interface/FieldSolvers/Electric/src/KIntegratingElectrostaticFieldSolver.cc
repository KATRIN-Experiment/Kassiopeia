/*
 * KIntegratingElectrostaticFieldSolver.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#include "KIntegratingElectrostaticFieldSolver.hh"
#include "KEMCout.hh"

namespace KEMField {

KIntegratingElectrostaticFieldSolver::KIntegratingElectrostaticFieldSolver() :
            		fIntegrator( NULL ),
					fIntegratingFieldSolver( NULL ),
#ifdef KEMFIELD_USE_OPENCL
					fOCLIntegrator( NULL ),
					fOCLIntegratingFieldSolver( NULL ),
#endif
					fUseOpenCL( false )
{
}

KIntegratingElectrostaticFieldSolver::~KIntegratingElectrostaticFieldSolver()
{
	delete fIntegrator;
	delete fIntegratingFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
	delete fOCLIntegrator;
	delete fOCLIntegratingFieldSolver;

	if( fUseOpenCL )
	{
		KOpenCLSurfaceContainer* oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( KOpenCLInterface::GetInstance()->GetActiveData() );
		if( oclContainer )
			delete oclContainer;
		oclContainer = NULL;
		KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
	}
#endif
}

void KIntegratingElectrostaticFieldSolver::InitializeCore( KSurfaceContainer& container )
{
	if( fUseOpenCL )
	{
#ifdef KEMFIELD_USE_OPENCL
		//KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
		KOpenCLSurfaceContainer* oclContainer;
		//if( data ) // this reuse of old data triggered openCL errors possibly due to changing workgroup sizes
		//	oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
		//else
		//{
			oclContainer = new KOpenCLSurfaceContainer( container );
			KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
		//}
		fOCLIntegrator = new KOpenCLElectrostaticBoundaryIntegrator{
				fIntegratorPolicy.CreateOpenCLConfig(),
				*oclContainer };
		fOCLIntegratingFieldSolver = new KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >( *oclContainer, *fOCLIntegrator );
		fOCLIntegratingFieldSolver->Initialize();
		return;
#else
		cout << "Warning: OpenCL not installed, running integrating field solver on CPU." << endl;
#endif
	}
	fIntegrator = new KElectrostaticBoundaryIntegrator{fIntegratorPolicy.CreateIntegrator()};
	fIntegratingFieldSolver = new KIntegratingFieldSolver< KElectrostaticBoundaryIntegrator >( container, *fIntegrator );
}

double KIntegratingElectrostaticFieldSolver::PotentialCore( const KPosition& P ) const
{
	if( fUseOpenCL )
	{
#ifdef KEMFIELD_USE_OPENCL
		return fOCLIntegratingFieldSolver->Potential( P );
#endif
	}
	return fIntegratingFieldSolver->Potential( P );
}

KEMThreeVector KIntegratingElectrostaticFieldSolver::ElectricFieldCore( const KPosition& P ) const
{
	if( fUseOpenCL )
	{
#ifdef KEMFIELD_USE_OPENCL
		return fOCLIntegratingFieldSolver->ElectricField( P );
#endif
	}
	return fIntegratingFieldSolver->ElectricField( P );
}

} //KEMField

