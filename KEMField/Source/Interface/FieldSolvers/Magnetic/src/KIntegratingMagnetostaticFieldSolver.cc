/*
 * KIntegratingMagnetostaticFieldSolver.cc
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#include "KIntegratingMagnetostaticFieldSolver.hh"

namespace KEMField
{


KIntegratingMagnetostaticFieldSolver::KIntegratingMagnetostaticFieldSolver() = default;

void KIntegratingMagnetostaticFieldSolver::InitializeCore(KElectromagnetContainer& container)
{
    if (!fIntegratingFieldSolver.Null())
        return;
    fIntegratingFieldSolver = new KIntegratingFieldSolver<KElectromagnetIntegrator>(container, fIntegrator);
}

KFieldVector KIntegratingMagnetostaticFieldSolver::MagneticPotentialCore(const KPosition& P) const
{
    return fIntegratingFieldSolver->VectorPotential(P);
}

KFieldVector KIntegratingMagnetostaticFieldSolver::MagneticFieldCore(const KPosition& P) const
{
    return fIntegratingFieldSolver->MagneticField(P);
}

KGradient KIntegratingMagnetostaticFieldSolver::MagneticGradientCore(const KPosition& P) const
{
    return fIntegratingFieldSolver->MagneticFieldGradient(P);
}

} /* namespace KEMField */
