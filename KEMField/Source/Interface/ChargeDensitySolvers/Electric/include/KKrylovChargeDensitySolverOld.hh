/*
 * KKrylovChargeDensitySolverOld.hh
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVCHARGEDENSITYSOLVEROLD_HH_
#define KKRYLOVCHARGEDENSITYSOLVEROLD_HH_

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

#endif /* KKRYLOVCHARGEDENSITYSOLVEROLD_HH_ */
