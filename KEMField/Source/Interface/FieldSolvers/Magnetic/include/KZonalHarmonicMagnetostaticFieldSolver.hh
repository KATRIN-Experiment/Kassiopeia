/*
 * KZonalHarmonicMagnetostaticFieldSolver.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_
#define KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_

#include "KMagneticFieldSolver.hh"

#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"

namespace KEMField {

class KZonalHarmonicMagnetostaticFieldSolver : public KMagneticFieldSolver
{
public:
    KZonalHarmonicMagnetostaticFieldSolver();
    virtual ~KZonalHarmonicMagnetostaticFieldSolver();

    void InitializeCore( KElectromagnetContainer& container );

    KThreeVector MagneticPotentialCore( const KPosition& P ) const;
    KThreeVector MagneticFieldCore( const KPosition& P ) const;
    KGradient MagneticGradientCore( const KPosition& P ) const;
    std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore( const KPosition& P ) const;

    KZonalHarmonicParameters* GetParameters()
    {
        return fParameters;
    }

private:
    KElectromagnetIntegrator fIntegrator;
    KZonalHarmonicContainer< KMagnetostaticBasis >* fZHContainer;
    KZonalHarmonicFieldSolver< KMagnetostaticBasis >* fZonalHarmonicFieldSolver;
    KZonalHarmonicParameters* fParameters;
};

} /* namespace KEMField */

#endif /* KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_ */
