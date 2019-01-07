/*
 * KGaussianEliminationChargeDensitySolver.hh
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_
#define KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

namespace KEMField {

class KGaussianEliminationChargeDensitySolver :
    public KChargeDensitySolver
{
    public:
        KGaussianEliminationChargeDensitySolver();
        virtual ~KGaussianEliminationChargeDensitySolver();

        virtual void InitializeCore( KSurfaceContainer& container );

        void SetIntegratorPolicy(const KEBIPolicy& integrator);
    private:
        KEBIPolicy fIntegratorPolicy;
};

} // KEMField

#endif /* KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_ */
