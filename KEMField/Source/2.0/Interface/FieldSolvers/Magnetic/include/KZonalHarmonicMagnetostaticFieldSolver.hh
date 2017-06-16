/*
 * KZonalHarmonicMagnetostaticFieldSolver.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_

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

    KEMThreeVector MagneticPotentialCore( const KPosition& P ) const;
    KEMThreeVector MagneticFieldCore( const KPosition& P ) const;
    KGradient MagneticGradientCore( const KPosition& P ) const;
    std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradientCore( const KPosition& P ) const;

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

#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_ */
