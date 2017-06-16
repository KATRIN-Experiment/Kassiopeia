/*
 * KSMagneticKEMField.h
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_
#define KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_

#include "KSMagneticField.h"

namespace KEMField {            // forward declaration has to be in namespace
    class KMagneticField;
} //KEMField

namespace Kassiopeia {

class KSMagneticKEMField: public KSMagneticField {
public:
    KSMagneticKEMField();
    KSMagneticKEMField(const KSMagneticKEMField& aCopy);
    KSMagneticKEMField(KEMField::KMagneticField* field);
    KSMagneticKEMField* Clone() const;
    virtual ~KSMagneticKEMField();

    void SetMagneticField(KEMField::KMagneticField* field);
    const KEMField::KMagneticField* getMagneticField();
    virtual void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aPotential);
    virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField);
    virtual void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient);
    virtual void CalculateFieldAndGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField, KThreeMatrix& aGradient);
private:
    void InitializeComponent();
    void DeinitializeComponent();

    KEMField::KMagneticField* fField;
};

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_FIELDS_INCLUDE_KSMAGNETICKEMFIELD_H_ */
