/*
 * KIntegratingMagnetostaticFieldSolver.hh
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#ifndef KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_
#define KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_

#include "KMagneticFieldSolver.hh"
#include "KElectromagnetIntegrator.hh"
#include "KElectromagnetIntegratingFieldSolver.hh"

#include "KSmartPointer.hh"

namespace KEMField {

class KIntegratingMagnetostaticFieldSolver: public KMagneticFieldSolver {

public:
    KIntegratingMagnetostaticFieldSolver();

    void InitializeCore( KElectromagnetContainer& container );

    KThreeVector MagneticPotentialCore( const KPosition& P ) const;
    KThreeVector MagneticFieldCore( const KPosition& P ) const;
    KGradient MagneticGradientCore( const KPosition& P ) const;

private:
    KElectromagnetIntegrator fIntegrator;
    KSmartPointer<KIntegratingFieldSolver< KElectromagnetIntegrator > > fIntegratingFieldSolver;
};

} /* namespace KEMField */

#endif /* KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_ */
