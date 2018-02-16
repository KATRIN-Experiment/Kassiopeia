/*
 * KKrylovChargeDensitySolverOld.hh
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVEROLD_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVEROLD_HH_

#include "KChargeDensitySolver.hh"

namespace KEMField {

class KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration;
class KFMElectrostaticFastMultipoleBoundaryValueSolver;

class KKrylovChargeDensitySolverOld : public KChargeDensitySolver {
public:
	KKrylovChargeDensitySolverOld();
	virtual ~KKrylovChargeDensitySolverOld();

	KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration*
	GetSolverConfig() {return fKrylovConfig;}

private:
	virtual void InitializeCore(KSurfaceContainer& container);

	KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration*
	fKrylovConfig;
	KFMElectrostaticFastMultipoleBoundaryValueSolver* fSolver;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVEROLD_HH_ */
