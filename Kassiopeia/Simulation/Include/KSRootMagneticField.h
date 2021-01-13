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
    void CalculatePotential(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                            KGeoBag::KThreeVector& aPotential) override;

    void CalculateField(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                        KGeoBag::KThreeVector& aField) override;

    void CalculateGradient(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                           KGeoBag::KThreeMatrix& aGradient) override;

    void CalculateFieldAndGradient(const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   KGeoBag::KThreeVector& aField, KGeoBag::KThreeMatrix& aGradient) override;

  public:
    void AddMagneticField(KSMagneticField* aMagneticField);
    void RemoveMagneticField(KSMagneticField* aMagneticField);

  private:
    KGeoBag::KThreeVector fCurrentPotential;
    KGeoBag::KThreeVector fCurrentField;
    KGeoBag::KThreeMatrix fCurrentGradient;

    KSList<KSMagneticField> fMagneticFields;

    static katrin::KGslErrorHandler& fGslErrorHandler;
};

}  // namespace Kassiopeia

#endif
