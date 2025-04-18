/*
 * KMagneticChargeDensitySolver.hh
 *
 *  Created on: 2 Apr 2025
 *      Author: pslocum
 */

#ifndef KMAGNETICCHARGEDENSITYSOLVER_HH_
#define KMAGNETICCHARGEDENSITYSOLVER_HH_

#include "KSurfaceContainer.hh"
#include "KThreeVector_KEMField.hh"

namespace KEMField
{

class KMagneticChargeDensitySolver
{
  public:
    KMagneticChargeDensitySolver() : fInitialized(false) {}
    virtual ~KMagneticChargeDensitySolver() = default;

    void Initialize(KSurfaceContainer& container)
    {
        if (!fInitialized) {
            InitializeCore(container);
            fInitialized = true;
        }
    }

  protected:
    virtual bool FindSolution();
    void SaveSolution(double threshold, KSurfaceContainer& container) const;

  private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;

    bool fInitialized;
};

}  // namespace KEMField

#endif /* KMAGNETICCHARGEDENSITYSOLVER_HH_ */
