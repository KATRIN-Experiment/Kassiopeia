/*
 * KSElectricKEMField.cxx
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */
#include "KSElectricKEMField.h"

#include "KElectricField.hh"

using namespace KEMField;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSElectricKEMField::KSElectricKEMField() : fField(nullptr) {}

KSElectricKEMField* KSElectricKEMField::Clone() const
{
    return new KSElectricKEMField(*this);
}

KSElectricKEMField::~KSElectricKEMField() = default;

KSElectricKEMField::KSElectricKEMField(const KSElectricKEMField& aCopy) : KSComponent(aCopy), fField(aCopy.fField) {}

KSElectricKEMField::KSElectricKEMField(KEMField::KElectricField* field) : KSComponent(), fField(field)
{
    SetName(field->GetName());
}

void KSElectricKEMField::SetElectricField(KEMField::KElectricField* field)
{
    fField = field;
}

KEMField::KElectricField* KSElectricKEMField::GetElectricField()
{
    return fField;
}

void KSElectricKEMField::CalculatePotential(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                            double& aPotential)
{
    aPotential = fField->Potential(aSamplePoint, aSampleTime);
}

void KSElectricKEMField::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                        KThreeVector& aField)
{
    aField = fField->ElectricField(aSamplePoint, aSampleTime);
}

void KSElectricKEMField::CalculateFieldAndPotential(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                                    KThreeVector& aField, double& aPotential)
{
    std::pair<KFieldVector, double> potential_field_pair = fField->ElectricFieldAndPotential(aSamplePoint, aSampleTime);
    aPotential = potential_field_pair.second;
    aField = potential_field_pair.first;
}

void KSElectricKEMField::InitializeComponent()
{
    fField->Initialize();
}

void KSElectricKEMField::DeinitializeComponent() {}

}  // namespace Kassiopeia
