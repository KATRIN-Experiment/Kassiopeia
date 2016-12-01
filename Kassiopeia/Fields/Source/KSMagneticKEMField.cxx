/*
 * KSMagneticKEMField.cpp
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#include "KSMagneticKEMField.h"
#include "KMagneticField.hh"
#include "KEMVectorConverters.hh"

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
    fName = field->Name();
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
    KEMThreeVector potential = fField->MagneticPotential(K2KEMThreeVector(aSamplePoint),aSampleTime);
    aPotential = KEM2KThreeVector(potential);
}

void KSMagneticKEMField::CalculateField(const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeVector& aField) {
    KEMThreeVector field = fField->MagneticField(K2KEMThreeVector(aSamplePoint),aSampleTime);
    aField = KEM2KThreeVector(field);
}

void KSMagneticKEMField::CalculateGradient(const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeMatrix& aGradient) {
    KGradient gradient  = fField->MagneticGradient(K2KEMThreeVector(aSamplePoint),aSampleTime);
    aGradient = KEM2KThreeMatrix(gradient);
}

void KSMagneticKEMField::CalculateFieldAndGradient( const KThreeVector& aSamplePoint,
        const double& aSampleTime, KThreeVector& aField, KThreeMatrix& aGradient)
{
    std::pair<KEMThreeVector, KGradient> field_gradient_pair = fField->MagneticFieldAndGradient(K2KEMThreeVector(aSamplePoint),aSampleTime);
    aField = field_gradient_pair.first;
    aGradient = field_gradient_pair.second;
}


void KSMagneticKEMField::InitializeComponent() {
    fField->Initialize();
}

void KSMagneticKEMField::DeinitializeComponent() {
}

} /* namespace Kassiopeia */
