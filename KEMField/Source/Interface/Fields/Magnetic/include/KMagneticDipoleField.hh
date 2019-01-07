/*
 * KMagneticDipoleField.hh
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICDIPOLEFIELD_HH_
#define KMAGNETICDIPOLEFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField {

class KMagneticDipoleField: public KMagnetostaticField {
public:
    KMagneticDipoleField();
    virtual ~KMagneticDipoleField();
private:
    KThreeVector MagneticPotentialCore( const KPosition& aSamplePoint ) const;
    KThreeVector MagneticFieldCore( const KPosition& aSamplePoint ) const;
    KGradient MagneticGradientCore( const KPosition& aSamplePoint) const;

public:
    void SetLocation( const KPosition& aLocation );
    void SetMoment( const KDirection& aMoment );

private:
    KPosition fLocation;
    KDirection fMoment;
};

} /* namespace KEMField */

#endif /* KMAGNETICDIPOLEFIELD_HH_ */
