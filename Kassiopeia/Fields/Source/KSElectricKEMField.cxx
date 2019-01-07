/*
 * KSElectricKEMField.cxx
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */
#include "KSElectricKEMField.h"
#include "KElectricField.hh"

using namespace KEMField;

namespace Kassiopeia {

KSElectricKEMField::KSElectricKEMField() : fField(NULL){
}

KSElectricKEMField::~KSElectricKEMField() {
}

KSElectricKEMField::KSElectricKEMField(const KSElectricKEMField& aCopy) :
		KSComponent(aCopy),fField(aCopy.fField)
{
}

KSElectricKEMField::KSElectricKEMField(KEMField::KElectricField* field) :
	KSComponent(),fField(field)
{
	SetName(field->Name());
}


KSElectricKEMField* KSElectricKEMField::Clone() const
{
	return new KSElectricKEMField( *this);
}

void KSElectricKEMField::CalculatePotential(
		const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
		double& aPotential) {
	aPotential = fField->Potential(aSamplePoint,aSampleTime);
}

void KSElectricKEMField::SetElectricField(KEMField::KElectricField* field) {
	fField = field;
}

const KEMField::KElectricField* KSElectricKEMField::getElectricField()
{
    return fField;
}

void KSElectricKEMField::CalculateField(
		const KGeoBag::KThreeVector& aSamplePoint, const double& aSampleTime,
		KGeoBag::KThreeVector& aField) {
	 aField = fField->ElectricField(aSamplePoint,aSampleTime);
}

void
KSElectricKEMField::CalculateFieldAndPotential( const KGeoBag::KThreeVector& aSamplePoint,
        const double& aSampleTime, KGeoBag::KThreeVector& aField, double& aPotential)
{
    std::pair<KThreeVector, double> potential_field_pair = fField->ElectricFieldAndPotential(aSamplePoint,aSampleTime);
    aPotential = potential_field_pair.second;
    aField = potential_field_pair.first;
}


void KSElectricKEMField::InitializeComponent() {
	fField->Initialize();
}

void KSElectricKEMField::DeinitializeComponent() {
}

} //Kassiopeia
