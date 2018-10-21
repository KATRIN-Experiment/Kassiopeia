/*
 * KMagneticDipoleField.hh
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField {

class KMagneticDipoleField: public KMagnetostaticField {
public:
    KMagneticDipoleField();
    virtual ~KMagneticDipoleField();
private:
    KEMThreeVector MagneticPotentialCore( const KPosition& aSamplePoint ) const;
    KEMThreeVector MagneticFieldCore( const KPosition& aSamplePoint ) const;
    KGradient MagneticGradientCore( const KPosition& aSamplePoint) const;

public:
    void SetLocation( const KPosition& aLocation );
    void SetMoment( const KDirection& aMoment );

private:
    KPosition fLocation;
    KDirection fMoment;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELD_HH_ */
