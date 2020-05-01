/*
 * KMagneticSuperpositionField.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KMagneticSuperpositionField.hh"

#include "KEMCout.hh"
#include "KMagnetostaticField.hh"

using namespace std;

namespace KEMField
{

KMagneticSuperpositionField::KMagneticSuperpositionField() :
    fMagneticFields(),
    fEnhancements(),
    fUseCaching(false),
    fCachingBlock(false),
    fPotentialCache(),
    fFieldCache(),
    fGradientCache()
{}

KMagneticSuperpositionField::~KMagneticSuperpositionField() {}

KThreeVector KMagneticSuperpositionField::MagneticPotentialCore(const KPosition& aSamplePoint,
                                                                const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedPotential(aSamplePoint, aSampleTime);

    return CalculateDirectPotential(aSamplePoint, aSampleTime);
}

KThreeVector KMagneticSuperpositionField::MagneticFieldCore(const KPosition& aSamplePoint,
                                                            const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedField(aSamplePoint, aSampleTime);

    return CalculateDirectField(aSamplePoint, aSampleTime);
}

KGradient KMagneticSuperpositionField::MagneticGradientCore(const KPosition& aSamplePoint,
                                                            const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if (fUseCaching && !fCachingBlock)
        return CalculateCachedGradient(aSamplePoint, aSampleTime);
    return CalculateDirectGradient(aSamplePoint, aSampleTime);
}

KThreeVector KMagneticSuperpositionField::CalculateCachedPotential(const KPosition& aSamplePoint,
                                                                   const double& aSampleTime) const
{
    KThreeVector aPotential(KThreeVector::sZero);
    //looking in cache for aSamplePoint
    auto potentialVector = fPotentialCache.find(aSamplePoint);
    if (potentialVector != fPotentialCache.end()) {
        for (size_t tIndex = 0; tIndex < potentialVector->second.size(); tIndex++) {
            aPotential += potentialVector->second.at(tIndex) * fEnhancements.at(tIndex);
        }
        return aPotential;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KThreeVector> tPotentials;
    KThreeVector tCurrentPotential;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        tCurrentPotential = fMagneticFields.at(tIndex)->MagneticPotential(aSamplePoint, aSampleTime);
        aPotential += tCurrentPotential * fEnhancements.at(tIndex);
        tPotentials.push_back(tCurrentPotential);
    }
    fPotentialCache.insert(make_pair(aSamplePoint, tPotentials));
    return aPotential;
}

KThreeVector KMagneticSuperpositionField::CalculateCachedField(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KThreeVector aField(KThreeVector::sZero);
    //looking in cache for aSamplePoint
    auto fieldVector = fFieldCache.find(aSamplePoint);
    if (fieldVector != fFieldCache.end()) {
        for (size_t tIndex = 0; tIndex < fieldVector->second.size(); tIndex++) {
            aField += fieldVector->second.at(tIndex) * fEnhancements.at(tIndex);
        }
        return aField;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KThreeVector> tFields;
    KThreeVector tCurrentField;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
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
        tCurrentGradient = fMagneticFields.at(tIndex)->MagneticGradient(aSamplePoint, aSampleTime);
        aGradient += tCurrentGradient * fEnhancements.at(tIndex);
        tGradients.push_back(tCurrentGradient);
    }
    fGradientCache.insert(make_pair(aSamplePoint, tGradients));
    return aGradient;
}

void KMagneticSuperpositionField::SetEnhancements(std::vector<double> aEnhancementVector)
{
    if (aEnhancementVector.size() != fEnhancements.size()) {
        KEMField::cout << "EnhancementVector has not the same size <" << aEnhancementVector.size()
                       << "> as fEnhancements <" << fEnhancements.size() << ">" << KEMField::endl;
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
    for (auto field : fMagneticFields) {
        field->Initialize();
    }
}

KThreeVector KMagneticSuperpositionField::CalculateDirectPotential(const KPosition& aSamplePoint,
                                                                   const double& aSampleTime) const
{
    KThreeVector potential(KThreeVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        potential +=
            fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticPotential(aSamplePoint, aSampleTime);
    }
    return potential;
}

KThreeVector KMagneticSuperpositionField::CalculateDirectField(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KThreeVector field(KThreeVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        field += fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticField(aSamplePoint, aSampleTime);
    }
    return field;
}

KGradient KMagneticSuperpositionField::CalculateDirectGradient(const KPosition& aSamplePoint,
                                                               const double& aSampleTime) const
{
    KGradient gradient(KGradient::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        gradient += fEnhancements.at(tIndex) * fMagneticFields.at(tIndex)->MagneticGradient(aSamplePoint, aSampleTime);
    }
    return gradient;
}

bool KMagneticSuperpositionField::AreAllFieldsStatic()
{
    bool allStatic(true);
    for (auto field : fMagneticFields) {
        if (!dynamic_cast<KMagnetostaticField*>(field))
            allStatic = false;
    }
    return allStatic;
}

void KMagneticSuperpositionField::CheckAndPrintCachingDisabledWarning() const
{
    if (fCachingBlock && fUseCaching) {
        KEMField::cout << "MagneticSuperpostionField supports caching only with static fields."
                          " Caching is disabled."
                       << KEMField::endl;
    }
}

} /* namespace KEMField */
