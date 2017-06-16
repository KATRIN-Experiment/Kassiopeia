/*
 * KMagneticFieldSolver.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KMAGNETICFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KMAGNETICFIELDSOLVER_HH_

#include "KElectromagnetContainer.hh"
#include "KEMThreeVector.hh"
#include "KEMThreeMatrix.hh"

namespace KEMField {

class KMagneticFieldSolver
{
public:
    KMagneticFieldSolver() : fInitialized(false) {}
    virtual ~KMagneticFieldSolver() {}

    void Initialize( KElectromagnetContainer& container) {
        if(!fInitialized) {
            InitializeCore(container);
            fInitialized = true;
        }
    }

    KEMThreeVector MagneticPotential ( const KPosition& P ) const {
        return MagneticPotentialCore(P);
    }

    KEMThreeVector MagneticField (const KPosition& P ) const {
        return MagneticFieldCore(P);
    }

    KGradient MagneticGradient( const KPosition& P ) const {
        return MagneticGradientCore(P);
    }

    std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradient( const KPosition& P ) const {
        return MagneticFieldAndGradientCore(P);
    }

private:

    virtual void InitializeCore(KElectromagnetContainer& container ) = 0;

    virtual KEMThreeVector MagneticPotentialCore ( const KPosition& P ) const = 0;
    virtual KEMThreeVector MagneticFieldCore (const KPosition& P ) const = 0;
    virtual KGradient MagneticGradientCore (const KPosition& P ) const = 0;

    virtual std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradientCore( const KPosition& P ) const {
        return std::make_pair(MagneticFieldCore(P),MagneticGradientCore(P));
    }

    bool fInitialized;

};


} /* KEMField */



#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_MAGNETIC_INCLUDE_KMAGNETICFIELDSOLVER_HH_ */
