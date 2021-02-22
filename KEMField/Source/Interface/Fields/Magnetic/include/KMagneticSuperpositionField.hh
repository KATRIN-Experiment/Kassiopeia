/*
 * KMagneticSuperpositionField.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICSUPERPOSITIONFIELD_HH_
#define KMAGNETICSUPERPOSITIONFIELD_HH_

#include "KMagneticField.hh"

#include <map>
#include <vector>

namespace KEMField
{

class KMagneticSuperpositionField : public KMagneticField
{
  public:
    enum RequirementType
    {
        rtNone, // no requirements on fields
        rtAll,  // require all fields to be valid
        rtAny,  // require one or more fields to be valid
        rtOne   // require exactly one field to be valid
    };

  public:
    KMagneticSuperpositionField();
    ~KMagneticSuperpositionField() override;

    bool CheckCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;

    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint, const double& aSampleTime) const override;

    void SetEnhancements(const std::vector<double>& aEnhancementVector);
    std::vector<double> GetEnhancements();

    void AddMagneticField(KMagneticField* aField, double aEnhancement = 1.0);

    void SetUseCaching(bool useCaching)
    {
        fUseCaching = useCaching;
    }

    void SetRequire(const std::string& require);

  private:
    void InitializeCore() override;

    KFieldVector CalculateCachedPotential(const KPosition& aSamplePoint, const double& aSampleTime) const;
    KFieldVector CalculateCachedField(const KPosition& aSamplePoint, const double& aSampleTime) const;
    KGradient CalculateCachedGradient(const KPosition& aSamplePoint, const double& aSampleTime) const;

    KFieldVector CalculateDirectPotential(const KPosition& aSamplePoint, const double& aSampleTime) const;
    KFieldVector CalculateDirectField(const KPosition& aSamplePoint, const double& aSampleTime) const;
    KGradient CalculateDirectGradient(const KPosition& aSamplePoint, const double& aSampleTime) const;

    bool AreAllFieldsStatic();
    void CheckAndPrintCachingDisabledWarning() const;

    std::vector<KMagneticField*> fMagneticFields;
    std::vector<double> fEnhancements;

    bool fUseCaching;
    bool fCachingBlock;
    RequirementType fRequire;
    mutable std::map<KPosition, std::vector<KFieldVector>> fPotentialCache;
    mutable std::map<KPosition, std::vector<KFieldVector>> fFieldCache;
    mutable std::map<KPosition, std::vector<KGradient>> fGradientCache;
};

} /* namespace KEMField */

#endif /* KMAGNETICSUPERPOSITIONFIELD_HH_ */
