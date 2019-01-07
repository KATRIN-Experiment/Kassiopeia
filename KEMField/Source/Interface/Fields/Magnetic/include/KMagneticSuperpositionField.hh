/*
 * KMagneticSuperpositionField.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICSUPERPOSITIONFIELD_HH_
#define KMAGNETICSUPERPOSITIONFIELD_HH_

#include "KMagneticField.hh"
#include <vector>
#include <map>

namespace KEMField {

class KMagneticSuperpositionField : public KMagneticField
{
public:
    KMagneticSuperpositionField();
    virtual ~KMagneticSuperpositionField();

    KThreeVector MagneticPotentialCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KThreeVector MagneticFieldCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient MagneticGradientCore( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    void SetEnhancements( std::vector< double > aEnhancementVector );
    std::vector< double > GetEnhancements();

    void AddMagneticField( KMagneticField* aField, double aEnhancement = 1.0 );

    void SetUseCaching( bool useCaching ) {fUseCaching = useCaching;}

private:
    void InitializeCore();

    KThreeVector CalculateCachedPotential( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KThreeVector CalculateCachedField( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient CalculateCachedGradient( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    KThreeVector CalculateDirectPotential( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KThreeVector CalculateDirectField( const KPosition& aSamplePoint, const double& aSampleTime ) const;
    KGradient CalculateDirectGradient( const KPosition& aSamplePoint, const double& aSampleTime ) const;

    bool AreAllFieldsStatic();
    void CheckAndPrintCachingDisabledWarning() const;

    std::vector< KMagneticField* > fMagneticFields;
    std::vector< double > fEnhancements;

    bool fUseCaching;
    bool fCachingBlock;
    mutable std::map < KPosition, std::vector<KThreeVector> > fPotentialCache;
    mutable std::map < KPosition, std::vector<KThreeVector> > fFieldCache;
    mutable std::map < KPosition, std::vector<KGradient> > fGradientCache;
};

} /* namespace KEMField */

#endif /* KMAGNETICSUPERPOSITIONFIELD_HH_ */
