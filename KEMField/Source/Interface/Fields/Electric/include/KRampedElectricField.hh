/*
 * KRampedElectricField.hh
 *
 *  Created on: 31 May 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KRAMPEDELECTRICFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KRAMPEDELECTRICFIELD_HH_

#include "KElectricField.hh"
#include "KField.h"

namespace KEMField
{

class KRampedElectricField: public KElectricField {
public:
    typedef enum {
        rtLinear,           // simple linear ramping
        rtExponential,      // exponential ramping with given time constant
        rtSinus,            // simple sinus ramping
    } eRampingType;

public:
    KRampedElectricField();
    virtual ~KRampedElectricField();

private:
    virtual double PotentialCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    virtual KEMThreeVector ElectricFieldCore( const KPosition& aSamplePoint, const double& aSampleTime) const;

public:
    double GetModulationFactor( const double& aTime ) const;

private:
    virtual void InitializeCore();

public:
    ;K_SET_GET_PTR( KElectricField, RootElectricField )
    ;K_SET_GET( eRampingType, RampingType )
    ;K_SET_GET( int, NumCycles )
    ;K_SET_GET( double, RampUpDelay )
    ;K_SET_GET( double, RampDownDelay )
    ;K_SET_GET( double, RampUpTime )
    ;K_SET_GET( double, RampDownTime )
    ;K_SET_GET( double, TimeConstant )
    ;K_SET_GET( double, TimeScalingFactor )
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KRAMPEDELECTRICFIELD_HH_ */
