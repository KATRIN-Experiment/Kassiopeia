/*
 * KSMagneticKEMField.cpp
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#include "KSMagneticKEMField.h"
#include "KMagneticField.hh"

using namespace KEMField;

namespace Kassiopeia {

KSMagneticKEMField::KSMagneticKEMField() :
        KSComponent(),fField(NULL)
{
}

KSMagneticKEMField::KSMagneticKEMField(const KSMagneticKEMField& aCopy) :
        KSComponent(aCopy),fField(aCopy.fField)
{
}


KSMagneticKEMField::KSMagneticKEMField(KEMField::KMagneticField* field) :
        KSComponent(),fField(field)
{
    SetName(field->Name());
}


KSMagneticKEMField* KSMagneticKEMField::Clone() const
{
    return new KSMagneticKEMField( *this);
}

KSMagneticKEMField::~KSMagneticKEMField()
{
}

void KSMagneticKEMField::SetMagneticField(KEMField::KMagneticField* field) {
    fField = field;
}

const KEMField::KMagneticField* KSMagneticKEMField::getMagneticField() {
    return fField;
}

void KSMagneticKEMField::CalculatePotential(const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeVector& aPotential) {
    aPotential = fField->MagneticPotential(aSamplePoint,aSampleTime);
}

void KSMagneticKEMField::CalculateField(const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeVector& aField) {
    aField = fField->MagneticField(aSamplePoint,aSampleTime);
}

void KSMagneticKEMField::CalculateGradient(const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeMatrix& aGradient) {
    aGradient  = fField->MagneticGradient(aSamplePoint,aSampleTime);
}

void KSMagneticKEMField::CalculateFieldAndGradient( const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeVector& aField, KThreeMatrix& aGradient)
{
    std::pair<KThreeVector, KGradient> field_gradient_pair = fField->MagneticFieldAndGradient(aSamplePoint,aSampleTime);
    aField = field_gradient_pair.first;
    aGradient = field_gradient_pair.second;
}


void KSMagneticKEMField::InitializeComponent() {
    fField->Initialize();
}

void KSMagneticKEMField::DeinitializeComponent() {
}

} /* namespace Kassiopeia */
