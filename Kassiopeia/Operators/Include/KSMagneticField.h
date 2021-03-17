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
    virtual void CalculatePotential(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                                    KGeoBag::KThreeVector& aPotential) = 0;

    virtual void CalculateField(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                                KGeoBag::KThreeVector& aField) = 0;

    virtual void CalculateGradient(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   KGeoBag::KThreeMatrix& aGradient) = 0;

    virtual void CalculateFieldAndGradient(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                                           KGeoBag::KThreeVector& aField, KGeoBag::KThreeMatrix& aGradient)
    {
        CalculateField(aSamplePoint, aSampleTime, aField);
        CalculateGradient(aSamplePoint, aSampleTime, aGradient);
    };
};

}  // namespace Kassiopeia

#endif
