/*
 * KMagneticSuperpositionField.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELD_HH_

#include "KMagneticField.hh"
#include <vector>
#include <map>

namespace KEMField {

class KMagneticSuperpositionField : public KMagneticField
{
public:
    KMagneticSuperpositionField();
    virtual ~KMagneticSuperpositionField();

    KEMThreeVector MagneticPotentialCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KEMThreeVector MagneticFieldCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient MagneticGradientCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    void SetEnhancements( std::vector< double > aEnhancementVector );
    std::vector< double > GetEnhancements();

    void AddMagneticField( KMagneticField* aField, double aEnhancement = 1.0 );

    void SetUseCaching( bool useCaching ) {fUseCaching = useCaching;}

private:
    void InitializeCore();

    KEMThreeVector CalculateCachedPotential( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KEMThreeVector CalculateCachedField( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient CalculateCachedGradient( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    KEMThreeVector CalculateDirectPotential( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KEMThreeVector CalculateDirectField( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient CalculateDirectGradient( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    bool AreAllFieldsStatic();
    void CheckAndPrintCachingDisabledWarning() const;

    std::vector< KMagneticField* > fMagneticFields;
    std::vector< double > fEnhancements;

    bool fUseCaching;
    bool fCachingBlock;
    mutable std::map < KPosition, std::vector<KEMThreeVector> > fPotentialCache;
    mutable std::map < KPosition, std::vector<KEMThreeVector> > fFieldCache;
    mutable std::map < KPosition, std::vector<KGradient> > fGradientCache;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELD_HH_ */
