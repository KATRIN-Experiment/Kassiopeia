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

    void SetHashProperties(unsigned int maskedBits, double hashThreshold);

  protected:
    virtual bool FindSolution(double threshold, KSurfaceContainer& container);
    void SaveSolution(double threshold, KSurfaceContainer& container) const;

  private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;

    unsigned int fHashMaskedBits;
    double fHashThreshold;
    bool fInitialized;
};

}  // namespace KEMField

#endif /* KMAGNETICCHARGEDENSITYSOLVER_HH_ */
