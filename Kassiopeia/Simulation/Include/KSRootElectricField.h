#ifndef Kassiopeia_KSRootElectricField_h_
#define Kassiopeia_KSRootElectricField_h_

#include "KGslErrorHandler.h"
#include "KSElectricField.h"
#include "KSList.h"

namespace Kassiopeia
{

class KSRootElectricField : public KSComponentTemplate<KSRootElectricField, KSElectricField>
{
  public:
    KSRootElectricField();
    KSRootElectricField(const KSRootElectricField& aCopy);
    KSRootElectricField* Clone() const override;
    ~KSRootElectricField() override;

  public:
    void CalculatePotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                            double& aPotential) override;

    void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                        katrin::KThreeVector& aField) override;

    void CalculateGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                           katrin::KThreeMatrix& aGradient) override;

    void CalculateFieldAndPotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                    katrin::KThreeVector& aField, double& aPotentia) override;

  public:
    void AddElectricField(KSElectricField* anElectricField);
    void RemoveElectricField(KSElectricField* anElectricField);

  private:
    double fCurrentPotential;
    katrin::KThreeVector fCurrentField;
    katrin::KThreeMatrix fCurrentGradient;

    KSList<KSElectricField> fElectricFields;

    static katrin::KGslErrorHandler& fGslErrorHandler;
};

}  // namespace Kassiopeia

#endif
