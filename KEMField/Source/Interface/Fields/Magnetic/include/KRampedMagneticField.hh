/*
 * KRampedMagneticField.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDMAGNETICFIELD_HH_
#define KRAMPEDMAGNETICFIELD_HH_

#include "KMagneticField.hh"

namespace KEMField
{

class KRampedMagneticField : public KMagneticField
{
  public:
    enum RampingType
    {
        rtLinear,       // simple linear ramping
        rtExponential,  // exponential ramping with given time constant
        rtInversion,    // ramping to inverted magnetic field using single exponential ramping
        rtInversion2,   // ramping to inverted magnetic field using exponential ramping with two time constants
        rtFlipBox       // ramping to inverted magnetic field using double exponential ramping (ramp to zero in between)
    };

  public:
    KRampedMagneticField();
    ~KRampedMagneticField() override;

  public:
    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;

  public:
    double GetModulationFactor(const double& aTime) const;
    double GetDerivModulationFactor(const double& aTime) const;

  protected:
    void InitializeCore() override;

  public:
    void SetMagneticField(KMagneticField* magneticField)
    {
        fRootMagneticField = magneticField;
    }
    const KMagneticField* GetMagneticField() const
    {
        return fRootMagneticField;
    }

    int GetNumCycles() const
    {
        return fNumCycles;
    }

    void SetNumCycles(int numCycles)
    {
        fNumCycles = numCycles;
    }

    double GetRampDownDelay() const
    {
        return fRampDownDelay;
    }

    void SetRampDownDelay(double rampDownDelay)
    {
        fRampDownDelay = rampDownDelay;
    }

    double GetRampDownTime() const
    {
        return fRampDownTime;
    }

    void SetRampDownTime(double rampDownTime)
    {
        fRampDownTime = rampDownTime;
    }

    RampingType GetRampingType() const
    {
        return fRampingType;
    }

    void SetRampingType(RampingType rampingType)
    {
        fRampingType = rampingType;
    }

    double GetRampUpDelay() const
    {
        return fRampUpDelay;
    }

    void SetRampUpDelay(double rampUpDelay)
    {
        fRampUpDelay = rampUpDelay;
    }

    double GetRampUpTime() const
    {
        return fRampUpTime;
    }

    void SetRampUpTime(double rampUpTime)
    {
        fRampUpTime = rampUpTime;
    }

    double GetTimeConstant() const
    {
        return fTimeConstant;
    }

    void SetTimeConstant(double timeConstant)
    {
        fTimeConstant = timeConstant;
    }

    double GetTimeConstant2() const
    {
        return fTimeConstant2;
    }

    void SetTimeConstant2(double timeConstant2)
    {
        fTimeConstant2 = timeConstant2;
    }

    double GetTimeScalingFactor() const
    {
        return fTimeScalingFactor;
    }

    void SetTimeScalingFactor(double timeScalingFactor)
    {
        fTimeScalingFactor = timeScalingFactor;
    }

  private:
    KMagneticField* fRootMagneticField;
    RampingType fRampingType;
    int fNumCycles;
    double fRampUpDelay;
    double fRampDownDelay;
    double fRampUpTime;
    double fRampDownTime;
    double fTimeConstant;
    double fTimeConstant2;
    double fTimeScalingFactor;
};

} /* namespace KEMField */

#endif /* KRAMPEDMAGNETICFIELD_HH_ */
