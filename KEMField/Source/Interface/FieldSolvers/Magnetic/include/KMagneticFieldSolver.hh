/*
 * KMagneticFieldSolver.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICFIELDSOLVER_HH_
#define KMAGNETICFIELDSOLVER_HH_

#include "KElectromagnetContainer.hh"
#include "KThreeMatrix_KEMField.hh"
#include "KThreeVector_KEMField.hh"

namespace KEMField
{

class KMagneticFieldSolver
{
  public:
    KMagneticFieldSolver() : fInitialized(false) {}
    virtual ~KMagneticFieldSolver() = default;

    void Initialize(KElectromagnetContainer& container)
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

    KFieldVector MagneticPotential(const KPosition& P) const
    {
        return MagneticPotentialCore(P);
    }

    KFieldVector MagneticField(const KPosition& P) const
    {
        return MagneticFieldCore(P);
    }

    KGradient MagneticGradient(const KPosition& P) const
    {
        return MagneticGradientCore(P);
    }

    std::pair<KFieldVector, KGradient> MagneticFieldAndGradient(const KPosition& P) const
    {
        return MagneticFieldAndGradientCore(P);
    }

  private:
    virtual void InitializeCore(KElectromagnetContainer& container) = 0;
    virtual void DeinitializeCore() = 0;

    virtual KFieldVector MagneticPotentialCore(const KPosition& P) const = 0;
    virtual KFieldVector MagneticFieldCore(const KPosition& P) const = 0;
    virtual KGradient MagneticGradientCore(const KPosition& P) const = 0;

    virtual std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const
    {
        return std::make_pair(MagneticFieldCore(P), MagneticGradientCore(P));
    }

    bool fInitialized;
};


}  // namespace KEMField


#endif /* KMAGNETICFIELDSOLVER_HH_ */
