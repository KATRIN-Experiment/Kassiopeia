/*
 * KGaussianEliminationChargeDensitySolver.hh
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_

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

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVER_HH_ */
