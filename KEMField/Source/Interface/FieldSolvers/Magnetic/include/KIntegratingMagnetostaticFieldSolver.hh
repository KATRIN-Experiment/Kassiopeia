/*
 * KIntegratingMagnetostaticFieldSolver.hh
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#ifndef KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_
#define KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_

#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KElectromagnetIntegrator.hh"
#include "KMagneticFieldSolver.hh"
#include "KSmartPointer.hh"

namespace KEMField
{

class KIntegratingMagnetostaticFieldSolver : public KMagneticFieldSolver
{

  public:
    KIntegratingMagnetostaticFieldSolver();

    void InitializeCore(KElectromagnetContainer& container) override;

    KThreeVector MagneticPotentialCore(const KPosition& P) const override;
    KThreeVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;

  private:
    KElectromagnetIntegrator fIntegrator;
    KSmartPointer<KIntegratingFieldSolver<KElectromagnetIntegrator>> fIntegratingFieldSolver;
};

} /* namespace KEMField */

#endif /* KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_ */
