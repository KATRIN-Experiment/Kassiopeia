/*
 * KIntegratingMagnetostaticFieldSolver.hh
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_

#include "KMagneticFieldSolver.hh"
#include "KElectromagnetIntegrator.hh"
#include "KElectromagnetIntegratingFieldSolver.hh"

#include "KSmartPointer.hh"

namespace KEMField {

class KIntegratingMagnetostaticFieldSolver: public KMagneticFieldSolver {

public:
    KIntegratingMagnetostaticFieldSolver();

    void InitializeCore( KElectromagnetContainer& container );

    KEMThreeVector MagneticPotentialCore( const KPosition& P ) const;
    KEMThreeVector MagneticFieldCore( const KPosition& P ) const;
    KGradient MagneticGradientCore( const KPosition& P ) const;

private:
    KElectromagnetIntegrator fIntegrator;
    KSmartPointer<KIntegratingFieldSolver< KElectromagnetIntegrator > > fIntegratingFieldSolver;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KINTEGRATINGMAGNETOSTATICFIELDSOLVER_HH_ */
