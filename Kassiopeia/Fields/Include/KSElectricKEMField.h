/*
 * KSElectricKEMField.h
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#ifndef KSELECTRICKEMFIELD_H_
#define KSELECTRICKEMFIELD_H_

#include "KSElectricField.h"

namespace KEMField {			// forward declaration has to be in namespace
	class KElectricField;
} // KEMField

namespace Kassiopeia {

class KSElectricKEMField : public KSElectricField {

public:
	KSElectricKEMField();
	KSElectricKEMField(const KSElectricKEMField& aCopy);
    KSElectricKEMField(KEMField::KElectricField* field);
	KSElectricKEMField* Clone() const;
	virtual ~KSElectricKEMField();

	void SetElectricField(KEMField::KElectricField* field);
	const KEMField::KElectricField* getElectricField();

	virtual void CalculatePotential( const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
	virtual void CalculateField( const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime, KGeoBag::KThreeVector& aField );
    virtual void CalculateFieldAndPotential( const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime, KGeoBag::KThreeVector& aField, double& aPotential);

private:
    void InitializeComponent();
    void DeinitializeComponent();


	KEMField::KElectricField* fField;
};

} // Kassiopeia

#endif /* KSELECTRICKEMFIELD_H_ */
