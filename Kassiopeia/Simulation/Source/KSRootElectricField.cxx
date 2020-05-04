#include "KSRootElectricField.h"

#include "KSException.h"
#include "KSFieldsMessage.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

using katrin::KGslErrorHandler;
KGslErrorHandler& KSRootElectricField::fGslErrorHandler = KGslErrorHandler::GetInstance();

KSRootElectricField::KSRootElectricField() :
    fCurrentPotential(),
    fCurrentField(),
    fCurrentGradient(),
    fElectricFields(128)
{
    fGslErrorHandler.Enable();
}
KSRootElectricField::KSRootElectricField(const KSRootElectricField& aCopy) :
    KSComponent(),
    fCurrentPotential(aCopy.fCurrentPotential),
    fCurrentField(aCopy.fCurrentField),
    fCurrentGradient(aCopy.fCurrentGradient),
    fElectricFields(aCopy.fElectricFields)
{
    fGslErrorHandler.Enable();
}
KSRootElectricField* KSRootElectricField::Clone() const
{
    return new KSRootElectricField(*this);
}
KSRootElectricField::~KSRootElectricField() {}

void KSRootElectricField::CalculatePotential(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                             double& aPotential)
{
    aPotential = 0.;
    try {
        for (int tIndex = 0; tIndex < fElectricFields.End(); tIndex++) {
            fElectricFields.ElementAt(tIndex)->CalculatePotential(aSamplePoint, aSampleTime, fCurrentPotential);
            aPotential += fCurrentPotential;
        }
    }
    catch (KSException const& e) {
        aPotential = numeric_limits<double>::quiet_NaN();
        throw KSFieldError().Nest(e) << "Failed to calculate electric potential at " << aSamplePoint << ".";
    }
    return;
}
void KSRootElectricField::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                         KThreeVector& aField)
{
    aField = KThreeVector::sZero;
    try {
        for (int tIndex = 0; tIndex < fElectricFields.End(); tIndex++) {
            fElectricFields.ElementAt(tIndex)->CalculateField(aSamplePoint, aSampleTime, fCurrentField);
            aField += fCurrentField;
        }
    }
    catch (KSException const& e) {
        aField = KThreeVector::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate electric field at " << aSamplePoint << ".";
    }
    return;
}
void KSRootElectricField::CalculateGradient(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                            KThreeMatrix& aGradient)
{
    aGradient = KThreeMatrix::sZero;
    try {
        for (int tIndex = 0; tIndex < fElectricFields.End(); tIndex++) {
            fElectricFields.ElementAt(tIndex)->CalculateGradient(aSamplePoint, aSampleTime, fCurrentGradient);
            aGradient += fCurrentGradient;
        }
    }
    catch (KSException const& e) {
        aGradient = KThreeMatrix::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate electric field gradient at " << aSamplePoint << ".";
    }
    return;
}
void KSRootElectricField::CalculateFieldAndPotential(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                                     KThreeVector& aField, double& aPotential)
{
    aField = KThreeVector::sZero;
    aPotential = 0.;
    try {
        for (int tIndex = 0; tIndex < fElectricFields.End(); tIndex++) {
            fElectricFields.ElementAt(tIndex)->CalculateFieldAndPotential(aSamplePoint,
                                                                          aSampleTime,
                                                                          fCurrentField,
                                                                          fCurrentPotential);
            aField += fCurrentField;
            aPotential += fCurrentPotential;
        }
    }
    catch (KSException const& e) {
        aField = KThreeVector::sInvalid;
        aPotential = numeric_limits<double>::quiet_NaN();
        throw KSFieldError().Nest(e) << "Failed to calculate electric field and potential at " << aSamplePoint << ".";
    }
    return;
}

void KSRootElectricField::AddElectricField(KSElectricField* anElectricField)
{
    //check that field is not already present
    for (int tIndex = 0; tIndex < fElectricFields.End(); tIndex++) {
        if (anElectricField == fElectricFields.ElementAt(tIndex)) {
            fieldmsg_debug("<" << GetName() << "> attempted to add electric field <" << anElectricField->GetName()
                               << "> which is already present." << eom);
            return;
        }
    }

    if (fElectricFields.AddElement(anElectricField) == -1) {
        fieldmsg(eError) << "<" << GetName() << "> could not add electric field <" << anElectricField->GetName() << ">"
                         << eom;
        return;
    }
    fieldmsg_debug("<" << GetName() << "> adding electric field <" << anElectricField->GetName() << ">" << eom);
    return;
}
void KSRootElectricField::RemoveElectricField(KSElectricField* anElectricField)
{
    if (fElectricFields.RemoveElement(anElectricField) == -1) {
        fieldmsg(eError) << "<" << GetName() << "> could not remove electric field <" << anElectricField->GetName()
                         << ">" << eom;
        return;
    }
    fieldmsg_debug("<" << GetName() << "> removing electric field <" << anElectricField->GetName() << ">" << eom);
    return;
}

STATICINT sKSRootElectricFieldDict = KSDictionary<KSRootElectricField>::AddCommand(
    &KSRootElectricField::AddElectricField, &KSRootElectricField::RemoveElectricField, "add_electric_field",
    "remove_electric_field");

}  // namespace Kassiopeia
