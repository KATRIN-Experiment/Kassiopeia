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
    void CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField) override;
    void CalculateGradient(const KThreeVector& aSamplePoint, const double& aSampleTime,
                           KThreeMatrix& aGradient) override;
    void CalculateFieldAndGradient(const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField,
                                   KThreeMatrix& aGradient) override;

  public:
    void AddMagneticField(KSMagneticField* aMagneticField);
    void RemoveMagneticField(KSMagneticField* aMagneticField);

  private:
    KThreeVector fCurrentField;
    KThreeMatrix fCurrentGradient;

    KSList<KSMagneticField> fMagneticFields;

    static katrin::KGslErrorHandler& fGslErrorHandler;
};

}  // namespace Kassiopeia

#endif
