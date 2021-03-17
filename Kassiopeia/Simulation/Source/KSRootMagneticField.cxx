#include "KSRootMagneticField.h"

#include "KSException.h"
#include "KSFieldsMessage.h"

using namespace KGeoBag;

namespace Kassiopeia
{
using katrin::KGslErrorHandler;
KGslErrorHandler& KSRootMagneticField::fGslErrorHandler = KGslErrorHandler::GetInstance();

KSRootMagneticField::KSRootMagneticField() : fCurrentField(), fCurrentGradient(), fMagneticFields(128)
{
    fGslErrorHandler.Enable();
}
KSRootMagneticField::KSRootMagneticField(const KSRootMagneticField& aCopy) :
    KSComponent(aCopy),
    fCurrentField(aCopy.fCurrentField),
    fCurrentGradient(aCopy.fCurrentGradient),
    fMagneticFields(aCopy.fMagneticFields)
{
    fGslErrorHandler.Enable();
}
KSRootMagneticField* KSRootMagneticField::Clone() const
{
    return new KSRootMagneticField(*this);
}
KSRootMagneticField::~KSRootMagneticField() = default;

void KSRootMagneticField::CalculatePotential(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                             KThreeVector& aPotential)
{
    aPotential = KThreeVector::sZero;
    try {
        for (int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++) {
            fMagneticFields.ElementAt(tIndex)->CalculatePotential(aSamplePoint, aSampleTime, fCurrentPotential);
            aPotential += fCurrentPotential;
        }
    }
    catch (KSException const& e) {
        aPotential = KThreeVector::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate magnetic potential at " << aSamplePoint << ".";
    }
    return;
}
void KSRootMagneticField::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                         KThreeVector& aField)
{
    aField = KThreeVector::sZero;
    try {
        for (int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++) {
            fMagneticFields.ElementAt(tIndex)->CalculateField(aSamplePoint, aSampleTime, fCurrentField);
            aField += fCurrentField;
        }
    }
    catch (KSException const& e) {
        aField = KThreeVector::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate magnetic field at " << aSamplePoint << ".";
    }
    return;
}
void KSRootMagneticField::CalculateGradient(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                            KThreeMatrix& aGradient)
{
    aGradient = KThreeMatrix::sZero;
    try {
        for (int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++) {
            fMagneticFields.ElementAt(tIndex)->CalculateGradient(aSamplePoint, aSampleTime, fCurrentGradient);
            aGradient += fCurrentGradient;
        }
    }
    catch (KSException const& e) {
        aGradient = KThreeMatrix::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate magnetic field gradient at " << aSamplePoint << ".";
    }
    return;
}

void KSRootMagneticField::CalculateFieldAndGradient(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                                    KThreeVector& aField, KThreeMatrix& aGradient)
{
    aField = KThreeVector::sZero;
    aGradient = KThreeMatrix::sZero;
    try {
        for (int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++) {
            fMagneticFields.ElementAt(tIndex)->CalculateFieldAndGradient(aSamplePoint,
                                                                         aSampleTime,
                                                                         fCurrentField,
                                                                         fCurrentGradient);
            aField += fCurrentField;
            aGradient += fCurrentGradient;
        }
    }
    catch (KSException const& e) {
        aField = KThreeVector::sInvalid;
        aGradient = KThreeMatrix::sInvalid;
        throw KSFieldError().Nest(e) << "Failed to calculate magnetic field and gradient at " << aSamplePoint << ".";
    }
    return;
}

void KSRootMagneticField::AddMagneticField(KSMagneticField* aMagneticField)
{
    //check that field is not already present
    if (fMagneticFields.FindElement(aMagneticField) != -1) {
        fieldmsg(eWarning) << "<" << GetName() << "> attempted to add magnetic field <" << aMagneticField->GetName()
                           << "> which is already present." << eom;
        return;
    }
    if (fMagneticFields.AddElement(aMagneticField) == -1) {
        fieldmsg(eError) << "<" << GetName() << "> could not add magnetic field <" << aMagneticField->GetName() << ">"
                         << eom;
        return;
    }
    fieldmsg_debug("<" << GetName() << "> adding magnetic field <" << aMagneticField->GetName() << ">" << eom);
    return;
}
void KSRootMagneticField::RemoveMagneticField(KSMagneticField* aMagneticField)
{
    if (fMagneticFields.RemoveElement(aMagneticField) == -1) {
        fieldmsg(eWarning) << "<" << GetName() << "> could not remove magnetic field <" << aMagneticField->GetName()
                           << ">" << eom;
        return;
    }
    fieldmsg_debug("<" << GetName() << "> removing magnetic field <" << aMagneticField->GetName() << ">" << eom);
    return;
}

STATICINT sKSRootMagneticFieldDict = KSDictionary<KSRootMagneticField>::AddCommand(
    &KSRootMagneticField::AddMagneticField, &KSRootMagneticField::RemoveMagneticField, "add_magnetic_field",
    "remove_magnetic_field");

}  // namespace Kassiopeia
