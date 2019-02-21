/*
 * KElectricZHFieldSolver.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KELECTRICZHFIELDSOLVER_HH_
#define KELECTRICZHFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"

#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

namespace KEMField {

class KElectricZHFieldSolver : public KElectricFieldSolver
{
public:
	KElectricZHFieldSolver();
	virtual ~KElectricZHFieldSolver();

	bool UseCentralExpansion( const KPosition& P );
	bool UseRemoteExpansion( const KPosition& P );

	void SetIntegratorPolicy(const KEBIPolicy& policy){
		fIntegratorPolicy = policy;
	}

	KZonalHarmonicParameters* GetParameters()
	{
		return fParameters;
	}
private:
	void InitializeCore( KSurfaceContainer& container );

	double PotentialCore( const KPosition& P ) const;
	KThreeVector ElectricFieldCore( const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const;

	KEBIPolicy fIntegratorPolicy;
	KElectrostaticBoundaryIntegrator fIntegrator;
	KZonalHarmonicContainer< KElectrostaticBasis >* fZHContainer;
	KZonalHarmonicFieldSolver< KElectrostaticBasis >* fZonalHarmonicFieldSolver;
	KZonalHarmonicParameters* fParameters;
};

} /* namespace KEMField */

#endif /* KELECTRICZHFIELDSOLVER_HH_ */
