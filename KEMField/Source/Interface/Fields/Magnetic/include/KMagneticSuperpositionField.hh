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
    KMagneticSuperpositionField();
    ~KMagneticSuperpositionField() override;

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

    void SetRequire(const std::string& require)
    {
        if (require == "all") fRequireAll = true;
        else fRequireAll = false;
        if (require == "one") fRequireOne = true;
        else fRequireOne = false;
        if (require == "any") fRequireAny = true;
        else fRequireAny = false;
    }

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
    bool fRequireAll;
    bool fRequireOne;
    bool fRequireAny;
    mutable std::map<KPosition, std::vector<KFieldVector>> fPotentialCache;
    mutable std::map<KPosition, std::vector<KFieldVector>> fFieldCache;
    mutable std::map<KPosition, std::vector<KGradient>> fGradientCache;
};

} /* namespace KEMField */

#endif /* KMAGNETICSUPERPOSITIONFIELD_HH_ */
