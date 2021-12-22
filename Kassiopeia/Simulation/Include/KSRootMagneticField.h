#ifndef Kassiopeia_KSRootMagneticField_h_
#define Kassiopeia_KSRootMagneticField_h_

#include "KGslErrorHandler.h"
#include "KSList.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

class KSRootMagneticField : public KSComponentTemplate<KSRootMagneticField, KSMagneticField>
{
  public:
    KSRootMagneticField();
    KSRootMagneticField(const KSRootMagneticField& aCopy);
    KSRootMagneticField* Clone() const override;
    ~KSRootMagneticField() override;

  public:
    void CalculatePotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                            katrin::KThreeVector& aPotential) override;

    void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                        katrin::KThreeVector& aField) override;

    void CalculateGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                           katrin::KThreeMatrix& aGradient) override;

    void CalculateFieldAndGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   katrin::KThreeVector& aField, katrin::KThreeMatrix& aGradient) override;

  public:
    void AddMagneticField(KSMagneticField* aMagneticField);
    void RemoveMagneticField(KSMagneticField* aMagneticField);

  private:
    katrin::KThreeVector fCurrentPotential;
    katrin::KThreeVector fCurrentField;
    katrin::KThreeMatrix fCurrentGradient;

    KSList<KSMagneticField> fMagneticFields;

    static katrin::KGslErrorHandler& fGslErrorHandler;
};

}  // namespace Kassiopeia

#endif
