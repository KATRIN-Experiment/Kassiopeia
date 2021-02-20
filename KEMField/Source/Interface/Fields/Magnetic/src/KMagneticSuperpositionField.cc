/*
 * KMagneticSuperpositionField.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KMagneticSuperpositionField.hh"

#include "KEMCoreMessage.hh"
#include "KMagnetostaticField.hh"

using namespace std;

namespace KEMField
{

KMagneticSuperpositionField::KMagneticSuperpositionField() : fUseCaching(false), fCachingBlock(false)
{
  SetRequire("all");
}

KMagneticSuperpositionField::~KMagneticSuperpositionField() = default;

bool KMagneticSuperpositionField::CheckCore(const KPosition& aSamplePoint,
                                            const double& aSampleTime) const
{
    bool check_all = true;
    bool check_any = false;
    bool check_one = false;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        if (! fMagneticFields.at(tIndex)->Check(aSamplePoint, aSampleTime)) {
            check_all = false;
            continue;
        }
        check_one = ! check_one && ! check_any;
        check_any = true;
    }
    if (fRequireAll && ! check_all)
        kem_cout(eWarning) << "MagneticSuperpositionField: at least one field not available at point <"
                           << aSamplePoint.X() << "," << aSamplePoint.Y() << "," << aSamplePoint.Z() << ">" << eom;
    if (fRequireAny && ! check_any)
        kem_cout(eWarning) << "MagneticSuperpositionField: not any fields available at point <"
                           << aSamplePoint.X() << "," << aSamplePoint.Y() << "," << aSamplePoint.Z() << ">" << eom;
    if (fRequireOne && ! check_one)
        kem_cout(eWarning) << "MagneticSuperpositionField: not exactly one field available at point <"
                           << aSamplePoint.X() << "," << aSamplePoint.Y() << "," << aSamplePoint.Z() << ">" << eom;
    return (fRequireAll && check_all) || (fRequireAny && check_any) || (fRequireOne && check_one);
}

KFieldVector KMagneticSuperpositionField::MagneticPotentialCore(const KPosition& aSamplePoint,
                                                                const double& aSampleTime) const
{
    CheckCore(aSamplePoint, aSampleTime);
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedPotential(aSamplePoint, aSampleTime);

    return CalculateDirectPotential(aSamplePoint, aSampleTime);
}

KFieldVector KMagneticSuperpositionField::MagneticFieldCore(const KPosition& aSamplePoint,
                                                            const double& aSampleTime) const
{
    CheckCore(aSamplePoint, aSampleTime);
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedField(aSamplePoint, aSampleTime);

    return CalculateDirectField(aSamplePoint, aSampleTime);
}

KGradient KMagneticSuperpositionField::MagneticGradientCore(const KPosition& aSamplePoint,
                                                            const double& aSampleTime) const
{
    CheckCore(aSamplePoint, aSampleTime);
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedGradient(aSamplePoint, aSampleTime);
    return CalculateDirectGradient(aSamplePoint, aSampleTime);
}

KFieldVector KMagneticSuperpositionField::CalculateCachedPotential(const KPosition& aSamplePoint,
                                                                   const double& aSampleTime) const
{
    KFieldVector aPotential(KFieldVector::sZero);
    //looking in cache for aSamplePoint
    auto potentialVector = fPotentialCache.find(aSamplePoint);
    if (potentialVector != fPotentialCache.end()) {
        for (size_t tIndex = 0; tIndex < potentialVector->second.size(); tIndex++) {
            aPotential += potentialVector->second.at(tIndex) * fEnhancements.at(tIndex);
        }
        return aPotential;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KFieldVector> tPotentials;
    KFieldVector tCurrentPotential;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        tCurrentPotential = fMagneticFields.at(tIndex)->MagneticPotential(aSamplePoint, aSampleTime);
        aPotential += tCurrentPotential * fEnhancements.at(tIndex);
        tPotentials.push_back(tCurrentPotential);
    }
    fPotentialCache.insert(make_pair(aSamplePoint, tPotentials));
    return aPotential;
}

KFieldVector KMagneticSuperpositionField::CalculateCachedField(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KFieldVector aField(KFieldVector::sZero);
    //looking in cache for aSamplePoint
    auto fieldVector = fFieldCache.find(aSamplePoint);
    if (fieldVector != fFieldCache.end()) {
        for (size_t tIndex = 0; tIndex < fieldVector->second.size(); tIndex++) {
            aField += fieldVector->second.at(tIndex) * fEnhancements.at(tIndex);
        }
        return aField;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KFieldVector> tFields;
    KFieldVector tCurrentField;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        if (! fMagneticFields.at(tIndex)->Check(aSamplePoint, aSampleTime)) continue;
        tCurrentField = fMagneticFields.at(tIndex)->MagneticField(aSamplePoint, aSampleTime);
        aField += tCurrentField * fEnhancements.at(tIndex);
        tFields.push_back(tCurrentField);
    }
    fFieldCache.insert(make_pair(aSamplePoint, tFields));
    return aField;
}

KGradient KMagneticSuperpositionField::CalculateCachedGradient(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KGradient aGradient(KGradient::sZero);
    //looking in cache for aSamplePoint
    auto gradientVector = fGradientCache.find(aSamplePoint);
    if (gradientVector != fGradientCache.end()) {
        for (size_t tIndex = 0; tIndex < gradientVector->second.size(); tIndex++) {
            aGradient += gradientVector->second.at(tIndex) * fEnhancements.at(tIndex);
        }
        return aGradient;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KGradient> tGradients;
    KGradient tCurrentGradient;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        if (! fMagneticFields.at(tIndex)->Check(aSamplePoint, aSampleTime)) continue;
        tCurrentGradient = fMagneticFields.at(tIndex)->MagneticGradient(aSamplePoint, aSampleTime);
        aGradient += tCurrentGradient * fEnhancements.at(tIndex);
        tGradients.push_back(tCurrentGradient);
    }
    fGradientCache.insert(make_pair(aSamplePoint, tGradients));
    return aGradient;
}

void KMagneticSuperpositionField::SetEnhancements(const std::vector<double>& aEnhancementVector)
{
    if (aEnhancementVector.size() != fEnhancements.size()) {
        kem_cout(eError) << "EnhancementVector has not the same size <" << aEnhancementVector.size()
                         << "> as fEnhancements <" << fEnhancements.size() << ">" << eom;
        exit(-1);
    }
    fEnhancements = aEnhancementVector;
    return;
}

std::vector<double> KMagneticSuperpositionField::GetEnhancements()
{
    return fEnhancements;
}

void KMagneticSuperpositionField::AddMagneticField(KMagneticField* aField, double aEnhancement)
{
    fMagneticFields.push_back(aField);
    fEnhancements.push_back(aEnhancement);
    fCachingBlock = !AreAllFieldsStatic();
    return;
}

void KMagneticSuperpositionField::InitializeCore()
{
    for (auto* field : fMagneticFields) {
        field->Initialize();
    }
}

KFieldVector KMagneticSuperpositionField::CalculateDirectPotential(const KPosition& aSamplePoint,
                                                                   const double& aSampleTime) const
{
    KFieldVector potential(KFieldVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        potential +=
            fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticPotential(aSamplePoint, aSampleTime);
    }
    return potential;
}

KFieldVector KMagneticSuperpositionField::CalculateDirectField(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KFieldVector field(KFieldVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        if (! fMagneticFields.at(tIndex)->Check(aSamplePoint, aSampleTime)) continue;
        field += fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticField(aSamplePoint, aSampleTime);
    }
    return field;
}

KGradient KMagneticSuperpositionField::CalculateDirectGradient(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KGradient gradient(KGradient::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        if (! fMagneticFields.at(tIndex)->Check(aSamplePoint, aSampleTime)) continue;
        gradient += fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticGradient(aSamplePoint, aSampleTime);
    }
    return gradient;
}

bool KMagneticSuperpositionField::AreAllFieldsStatic()
{
    bool allStatic(true);
    for (auto* field : fMagneticFields) {
        if (!dynamic_cast<KMagnetostaticField*>(field))
            allStatic = false;
    }
    return allStatic;
}

void KMagneticSuperpositionField::CheckAndPrintCachingDisabledWarning() const
{
    if (fCachingBlock && fUseCaching) {
        kem_cout(eWarning) << "MagneticSuperpostionField supports caching only with static fields."
                              " Caching is disabled."
                           << eom;
    }
}

} /* namespace KEMField */
