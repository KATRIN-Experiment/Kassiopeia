/*
 * KRampedElectricField.hh
 *
 *  Created on: 31 May 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDELECTRICFIELD_HH_
#define KRAMPEDELECTRICFIELD_HH_

#include "KElectricField.hh"
#include "KField.h"

namespace KEMField
{

class KRampedElectricField : public KElectricField
{
  public:
    typedef enum
    {
        rtLinear,       // simple linear ramping
        rtExponential,  // exponential ramping with given time constant
        rtSinus,        // simple sinus ramping
    } eRampingType;

  public:
    KRampedElectricField();
    ~KRampedElectricField() override;

  private:
    double PotentialCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;
    KFieldVector ElectricFieldCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;

  public:
    double GetModulationFactor(const double& aTime) const;

  private:
    void InitializeCore() override;

  public:
    ;
    K_SET_GET_PTR(KElectricField, RootElectricField);
    K_SET_GET(eRampingType, RampingType);
    K_SET_GET(int, NumCycles);
    K_SET_GET(double, RampUpDelay);
    K_SET_GET(double, RampDownDelay);
    K_SET_GET(double, RampUpTime);
    K_SET_GET(double, RampDownTime);
    K_SET_GET(double, TimeConstant);
    K_SET_GET(double, TimeScalingFactor)
};

} /* namespace KEMField */

#endif /* KRAMPEDELECTRICFIELD_HH_ */
