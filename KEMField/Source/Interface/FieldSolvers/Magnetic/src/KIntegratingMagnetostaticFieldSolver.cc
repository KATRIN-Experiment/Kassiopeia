/*
 * KIntegratingMagnetostaticFieldSolver.cc
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#include "KIntegratingMagnetostaticFieldSolver.hh"

namespace KEMField {



KIntegratingMagnetostaticFieldSolver::KIntegratingMagnetostaticFieldSolver() {}

void KIntegratingMagnetostaticFieldSolver::InitializeCore(
        KElectromagnetContainer& container)
{
    if (!fIntegratingFieldSolver.Null()) return;
    fIntegratingFieldSolver =
            new KIntegratingFieldSolver< KElectromagnetIntegrator >(
                    container,
                    fIntegrator);
}

KThreeVector KIntegratingMagnetostaticFieldSolver::MagneticPotentialCore(
        const KPosition& P) const {
    return fIntegratingFieldSolver->VectorPotential( P );
}

KThreeVector KIntegratingMagnetostaticFieldSolver::MagneticFieldCore(
        const KPosition& P) const {
    return fIntegratingFieldSolver->MagneticField( P );
}

KGradient KIntegratingMagnetostaticFieldSolver::MagneticGradientCore(
        const KPosition& P) const {
    return fIntegratingFieldSolver->MagneticFieldGradient( P );
}

} /* namespace KEMField */
