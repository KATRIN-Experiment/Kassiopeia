/*
 * KRampedElectric2Field.hh
 *
 *  Created on: 16 Jun 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDELECTRIC2FIELD_HH_
#define KRAMPEDELECTRIC2FIELD_HH_

#include "KField.h"
#include "KElectricField.hh"

namespace KEMField {

class KRampedElectric2Field : public KElectricField    {
public:
    typedef enum {
        rtLinear,           // simple linear ramping
        rtExponential,      // exponential ramping with given time constant
        rtSinus,            // simple sinus ramping
        rtSquare,           // simple square ramping
    } eRampingType;

public:
    KRampedElectric2Field();
    virtual ~KRampedElectric2Field();

private:
    virtual double PotentialCore( const KPosition& aSamplePoint, const double& aSampleTime) const;
    virtual KThreeVector ElectricFieldCore( const KPosition& aSamplePoint, const double& aSampleTime) const;

public:
    double GetModulationFactor( const double& aTime ) const;

private:
    virtual void InitializeCore();

public:
    ;K_SET_GET_PTR( KElectricField, RootElectricField1 )
    ;K_SET_GET_PTR( KElectricField, RootElectricField2 )
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

#endif /* KRAMPEDELECTRIC2FIELD_HH_ */
