#ifndef Kassiopeia_KSMagneticField_h_
#define Kassiopeia_KSMagneticField_h_

#include "KSComponentTemplate.h"

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSMagneticField : public KSComponentTemplate<KSMagneticField>
{
  public:
    KSMagneticField();
    ~KSMagneticField() override;

  public:
    virtual void CalculatePotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                    katrin::KThreeVector& aPotential) = 0;

    virtual void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                katrin::KThreeVector& aField) = 0;

    virtual void CalculateGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   katrin::KThreeMatrix& aGradient) = 0;

    virtual void CalculateFieldAndGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                           katrin::KThreeVector& aField, katrin::KThreeMatrix& aGradient)
    {
        CalculateField(aSamplePoint, aSampleTime, aField);
        CalculateGradient(aSamplePoint, aSampleTime, aGradient);
    };
};

}  // namespace Kassiopeia

#endif
