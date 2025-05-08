/*
 * KMagneticFFTMFieldSolver.hh
 *
 *  Created on: 07.05.2025
 *      Author: pslocum
 */

#ifndef KMAGNETICFFTMFIELDSOLVER_HH_
#define KMAGNETICFFTMFIELDSOLVER_HH_

#include "KSurfaceContainer.hh"
#include "KThreeVector_KEMField.hh"

#include <memory>

namespace KEMField
{

class KMagneticFFTMFieldSolver
{
  public:
    KMagneticFFTMFieldSolver() : fInitialized(false) {}
    virtual ~KMagneticFFTMFieldSolver() = default;

    void Initialize(KSurfaceContainer& container)
    {
        if (!fInitialized) {
            InitializeCore(container);
            fInitialized = true;
        }
    }
    void Deinitialize()
    {
        if (fInitialized) {
            DeinitializeCore();
            fInitialized = false;
        }
    }

    KFieldVector MagneticField(const KPosition& P) const
    {
        return MagneticFieldCore(P);
    }

  private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;
    virtual void DeinitializeCore() = 0;

    virtual KFieldVector MagneticFieldCore(const KPosition& P) const = 0;

    bool fInitialized;
};

}  // namespace KEMField

#endif /* KMAGNETICFFTMFIELDSOLVER_HH_ */
