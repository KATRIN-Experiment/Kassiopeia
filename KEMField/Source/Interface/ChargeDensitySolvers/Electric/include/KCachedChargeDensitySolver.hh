/*
 * KCachedChargeDensitySolver.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 *
 *      Imported from KSFieldElectrostatic
 */

#ifndef KCACHEDCHARGEDENSITYSOLVER_HH_
#define KCACHEDCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"

namespace KEMField
{

class KCachedChargeDensitySolver : public KChargeDensitySolver
{
  public:
    KCachedChargeDensitySolver();
    ~KCachedChargeDensitySolver() override;

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

#endif /* KCACHEDCHARGEDENSITYSOLVER_HH_ */
