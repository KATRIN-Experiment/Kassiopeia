#ifndef Kassiopeia_KSElectricField_h_
#define Kassiopeia_KSElectricField_h_

#include "KSComponentTemplate.h"

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSElectricField : public KSComponentTemplate<KSElectricField>
{
  public:
    KSElectricField();
    ~KSElectricField() override;

  public:
    virtual void CalculatePotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                    double& aPotential) = 0;

    virtual void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                katrin::KThreeVector& aField) = 0;

    virtual void CalculateGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   katrin::KThreeMatrix& aGradient);

    virtual void CalculateFieldAndPotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                            katrin::KThreeVector& aField, double& aPotential)
    {
        CalculateField(aSamplePoint, aSampleTime, aField);
        CalculatePotential(aSamplePoint, aSampleTime, aPotential);
    };
};

}  // namespace Kassiopeia

#endif
