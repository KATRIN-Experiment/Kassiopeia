/*
 * KInducedAzimuthalElectricField.hh
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#ifndef KINDUCEDAZIMUTHALELECTRICFIELD_HH_
#define KINDUCEDAZIMUTHALELECTRICFIELD_HH_

#include "KElectricField.hh"

namespace KEMField {

class KRampedMagneticField;

class KInducedAzimuthalElectricField : public KElectricField {
public:
    KInducedAzimuthalElectricField();

    void SetMagneticField(KRampedMagneticField* field);
    KRampedMagneticField* GetRampedMagneticField() const;


private:
    double PotentialCore( const KPosition& P,const double& time) const;
    KThreeVector ElectricFieldCore( const KPosition& P, const double& time) const;
    virtual void InitializeCore();

    void CheckMagneticField() const;

    KRampedMagneticField* fMagneticField;
};

} /* namespace KEMField */

#endif /* KINDUCEDAZIMUTHALELECTRICFIELD_HH_ */
