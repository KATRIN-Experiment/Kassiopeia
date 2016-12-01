/*
 * KInducedAzimuthalElectricField.hh
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KINDUCEDAZIMUTHALELECTRICFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KINDUCEDAZIMUTHALELECTRICFIELD_HH_

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
    KEMThreeVector ElectricFieldCore( const KPosition& P, const double& time) const;
    virtual void InitializeCore();

    void CheckMagneticField() const;

    KRampedMagneticField* fMagneticField;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KINDUCEDAZIMUTHALELECTRICFIELD_HH_ */
