/*
 * KElectricZHFieldSolver.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICZHFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICZHFIELDSOLVER_HH_

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
	KEMThreeVector ElectricFieldCore( const KPosition& P ) const;
    std::pair<KEMThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const;

	KEBIPolicy fIntegratorPolicy;
	KElectrostaticBoundaryIntegrator fIntegrator;
	KZonalHarmonicContainer< KElectrostaticBasis >* fZHContainer;
	KZonalHarmonicFieldSolver< KElectrostaticBasis >* fZonalHarmonicFieldSolver;
	KZonalHarmonicParameters* fParameters;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_INCLUDE_KELECTRICZHFIELDSOLVER_HH_ */
