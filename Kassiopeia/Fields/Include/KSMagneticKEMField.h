/*
 * KSMagneticKEMField.h
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_
#define KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_

#include "KSMagneticField.h"

namespace KEMField  // forward declaration has to be in namespace
{
class KMagneticField;
}  // namespace KEMField

namespace Kassiopeia
{

class KSMagneticKEMField : public KSMagneticField
{
  public:
    KSMagneticKEMField();
    KSMagneticKEMField(const KSMagneticKEMField& aCopy);
    KSMagneticKEMField(KEMField::KMagneticField* field);
    KSMagneticKEMField* Clone() const override;
    ~KSMagneticKEMField() override;

    void SetMagneticField(KEMField::KMagneticField* field);
    KEMField::KMagneticField* GetMagneticField();

    void CalculatePotential(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                            katrin::KThreeVector& aPotential) override;

    void CalculateField(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                        katrin::KThreeVector& aField) override;

    void CalculateGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                           katrin::KThreeMatrix& aGradient) override;

    void CalculateFieldAndGradient(const katrin::KThreeVector& aSamplePoint, const double& aSampleTime,
                                   katrin::KThreeVector& aField, katrin::KThreeMatrix& aGradient) override;

  private:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

    KEMField::KMagneticField* fField;
};

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_ */
