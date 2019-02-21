/*
 * KMagnetostaticConstantField.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICCONSTANTFIELD_HH_
#define KMAGNETOSTATICCONSTANTFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField {

class KMagnetostaticConstantField: public KEMField::KMagnetostaticField
{
public:
    KMagnetostaticConstantField() :
        KMagnetostaticField(),
        fFieldVector() {}

    KMagnetostaticConstantField( const KThreeVector& aField ) :
        KMagnetostaticField(),
        fFieldVector(aField) {}

    virtual ~KMagnetostaticConstantField() {}

private:
    /** We choose A(r) = 1/2 * B x r as the magnetic potential.
     * This is a viable choice for Coulomb gauge.*/
    virtual KThreeVector MagneticPotentialCore(const KPosition& P) const {
        return 0.5 * fFieldVector.Cross(P);
    }
    virtual KThreeVector MagneticFieldCore(const KPosition& /*P*/) const {
        return fFieldVector;
    }
    virtual KGradient MagneticGradientCore(const KPosition& /*P*/) const {
        return KThreeMatrix::sZero;
    }

public:
    void SetField( const KThreeVector& aFieldVector ) {
        fFieldVector = aFieldVector;
    }

    KThreeVector GetField() const {
        return fFieldVector;
    }

private:
    KThreeVector fFieldVector;
};

} /* namespace KEMFIELD */

#endif /* KMAGNETOSTATICCONSTANTFIELD_HH_ */
