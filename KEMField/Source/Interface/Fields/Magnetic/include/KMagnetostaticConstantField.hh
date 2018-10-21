/*
 * KMagnetostaticConstantField.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField {

class KMagnetostaticConstantField: public KEMField::KMagnetostaticField
{
public:
    KMagnetostaticConstantField() :
        KMagnetostaticField(),
        fFieldVector() {}

    KMagnetostaticConstantField( const KEMThreeVector& aField ) :
        KMagnetostaticField(),
        fFieldVector(aField) {}

    virtual ~KMagnetostaticConstantField() {}

private:
    /** We choose A(r) = 1/2 * B x r as the magnetic potential.
     * This is a viable choice for Coulomb gauge.*/
    virtual KEMThreeVector MagneticPotentialCore(const KPosition& P) const {
        return 0.5 * fFieldVector.Cross(P);
    }
    virtual KEMThreeVector MagneticFieldCore(const KPosition& /*P*/) const {
        return fFieldVector;
    }
    virtual KGradient MagneticGradientCore(const KPosition& /*P*/) const {
        return KEMThreeMatrix::sZero;
    }

public:
    void SetField( const KEMThreeVector& aFieldVector ) {
        fFieldVector = aFieldVector;
    }

    KEMThreeVector GetField() const {
        return fFieldVector;
    }

private:
    KEMThreeVector fFieldVector;
};

} /* namespace KEMFIELD */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELD_HH_ */
