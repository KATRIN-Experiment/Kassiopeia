/*
 * KCachedMagneticChargeDensitySolver.hh
 *
 *  Created on: 18 Apr 2025
 *      Author: pslocum
 *
 *      Imported from KSFieldMagnetostatic
 */

#ifndef KCACHEDMAGNETICCHARGEDENSITYSOLVER_HH_
#define KCACHEDMAGNETICCHARGEDENSITYSOLVER_HH_

#include "KMagneticChargeDensitySolver.hh"

namespace KEMField
{

class KCachedMagneticChargeDensitySolver : public KMagneticChargeDensitySolver
{
  public:
    KCachedMagneticChargeDensitySolver();
    ~KCachedMagneticChargeDensitySolver() override;

    void SetName(const std::string& s)
    {
        fName = s;
    }
    void SetHash(const std::string& s)
    {
        fHash = s;
    }

  private:
    void InitializeCore(KSurfaceContainer& container) override;

    std::string fName;
    std::string fHash;
};

}  // namespace KEMField

#endif /* KCACHEDMAGNETICCHARGEDENSITYSOLVER_HH_ */
