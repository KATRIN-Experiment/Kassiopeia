/*
 * KMagneticSuperpositionField.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KMagneticSuperpositionField.hh"
#include "KMagnetostaticField.hh"
#include "KEMCout.hh"

using namespace std;

namespace KEMField {

KMagneticSuperpositionField::KMagneticSuperpositionField()  :
                fMagneticFields(),
                fEnhancements(),
                fUseCaching( false ),
                fCachingBlock( false ),
                fPotentialCache(),
                fFieldCache(),
                fGradientCache(){
}

KMagneticSuperpositionField::~KMagneticSuperpositionField() {
}

KEMThreeVector KMagneticSuperpositionField::MagneticPotentialCore(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if ( fUseCaching && !fCachingBlock)
        return CalculateCachedPotential(aSamplePoint, aSampleTime);

    return CalculateDirectPotential(aSamplePoint, aSampleTime);
}

KEMThreeVector KMagneticSuperpositionField::MagneticFieldCore(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if( fUseCaching && !fCachingBlock)
        return CalculateCachedField(aSamplePoint, aSampleTime);

    return CalculateDirectField(aSamplePoint, aSampleTime);
}

KGradient KMagneticSuperpositionField::MagneticGradientCore(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    CheckAndPrintCachingDisabledWarning();
    if( fUseCaching && !fCachingBlock)
        return CalculateCachedGradient(aSamplePoint, aSampleTime);
    return CalculateDirectGradient(aSamplePoint,aSampleTime);

}

KEMThreeVector KMagneticSuperpositionField::CalculateCachedPotential(
        const KPosition& aSamplePoint, const double& aSampleTime) const {
    KEMThreeVector aPotential (KEMThreeVector::sZero);
    //looking in cache for aSamplePoint
    auto potentialVector = fPotentialCache.find(aSamplePoint);
    if(potentialVector != fPotentialCache.end() ) {
        for (size_t tIndex = 0; tIndex < potentialVector->second.size(); tIndex++)
        {
            aPotential += potentialVector->second.at( tIndex ) * fEnhancements.at( tIndex );
        }
        return aPotential;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KEMThreeVector> tPotentials;
    KEMThreeVector tCurrentPotential;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        tCurrentPotential = fMagneticFields.at( tIndex )->MagneticPotential( aSamplePoint, aSampleTime );
        aPotential += tCurrentPotential * fEnhancements.at( tIndex );
        tPotentials.push_back( tCurrentPotential );
    }
    fPotentialCache.insert( make_pair(aSamplePoint,tPotentials) );
    return aPotential;
}

KEMThreeVector KMagneticSuperpositionField::CalculateCachedField(
        const KPosition& aSamplePoint, const double& aSampleTime) const {
    KEMThreeVector aField (KEMThreeVector::sZero);
    //looking in cache for aSamplePoint
    auto fieldVector = fFieldCache.find(aSamplePoint);
    if(fieldVector != fFieldCache.end() ) {
        for (size_t tIndex = 0; tIndex < fieldVector->second.size(); tIndex++)
        {
            aField += fieldVector->second.at( tIndex ) * fEnhancements.at( tIndex );
        }
        return aField;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KEMThreeVector> tFields;
    KEMThreeVector tCurrentField;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        tCurrentField = fMagneticFields.at( tIndex )->MagneticField( aSamplePoint, aSampleTime );
        aField += tCurrentField * fEnhancements.at( tIndex );
        tFields.push_back( tCurrentField );
    }
    fFieldCache.insert( make_pair(aSamplePoint,tFields) );
    return aField;
}

KGradient KMagneticSuperpositionField::CalculateCachedGradient(
        const KPosition& aSamplePoint, const double& aSampleTime) const {
    KGradient aGradient (KGradient::sZero);
    //looking in cache for aSamplePoint
    auto gradientVector = fGradientCache.find(aSamplePoint);
    if(gradientVector != fGradientCache.end() ) {
        for (size_t tIndex = 0; tIndex < gradientVector->second.size(); tIndex++)
        {
            aGradient += gradientVector->second.at( tIndex ) * fEnhancements.at( tIndex );
        }
        return aGradient;
    }

    //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
    vector<KGradient> tGradients;
    KGradient tCurrentGradient;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        tCurrentGradient = fMagneticFields.at( tIndex )->MagneticGradient( aSamplePoint, aSampleTime );
        aGradient += tCurrentGradient * fEnhancements.at( tIndex );
        tGradients.push_back( tCurrentGradient );
    }
    fGradientCache.insert( make_pair(aSamplePoint,tGradients) );
    return aGradient;
}

void KMagneticSuperpositionField::SetEnhancements(
        std::vector<double> aEnhancementVector) {
    if ( aEnhancementVector.size() != fEnhancements.size() )
    {
        KEMField::cout <<"EnhancementVector has not the same size <"
                <<aEnhancementVector.size()<<"> as fEnhancements <"
                <<fEnhancements.size()<<">" << KEMField::endl;
        exit(-1);
    }
    fEnhancements = aEnhancementVector;
    return;
}

std::vector<double> KMagneticSuperpositionField::GetEnhancements() {
    return fEnhancements;
}

void KMagneticSuperpositionField::AddMagneticField(
        KMagneticField* aField, double aEnhancement) {
    fMagneticFields.push_back( aField );
    fEnhancements.push_back( aEnhancement );
    fCachingBlock = !AreAllFieldsStatic();
    return;
}

void KMagneticSuperpositionField::InitializeCore() {
    for (auto field : fMagneticFields )
    {
        field->Initialize();
    }
}

KEMThreeVector KMagneticSuperpositionField::CalculateDirectPotential(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KEMThreeVector potential(KEMThreeVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        potential += fEnhancements.at(tIndex) *
                fMagneticFields.at( tIndex )->MagneticPotential( aSamplePoint, aSampleTime);
    }
    return potential;
}

KEMThreeVector KMagneticSuperpositionField::CalculateDirectField(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KEMThreeVector field(KEMThreeVector::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        field += fEnhancements.at(tIndex) *
                fMagneticFields.at( tIndex )->MagneticField( aSamplePoint, aSampleTime);
    }
    return field;
}

KGradient KMagneticSuperpositionField::CalculateDirectGradient(
        const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KGradient gradient(KGradient::sZero);
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
    {
        gradient += fEnhancements.at(tIndex) *
                fMagneticFields.at( tIndex )->MagneticGradient( aSamplePoint, aSampleTime);
    }
    return gradient;
}

bool KMagneticSuperpositionField::AreAllFieldsStatic() {
    bool allStatic ( true );
    for (auto field : fMagneticFields ){
        if(!dynamic_cast<KMagnetostaticField*>(field))
            allStatic = false;
    }
    return allStatic;
}

void KMagneticSuperpositionField::CheckAndPrintCachingDisabledWarning() const
{
    if( fCachingBlock && fUseCaching )
    {
        KEMField::cout << "MagneticSuperpostionField supports caching only with static fields."
        " Caching is disabled." << KEMField::endl;
    }
}

} /* namespace KEMField */
