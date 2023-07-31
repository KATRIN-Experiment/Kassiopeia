/*
 * KChargeDensitySolver.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KELECTRICFIELDSOLVER_HH_
#define KELECTRICFIELDSOLVER_HH_

#include "KSurfaceContainer.hh"
#include "KThreeVector_KEMField.hh"

#include <memory>

namespace KEMField
{

class KElectricFieldSolver
{
  public:
    KElectricFieldSolver() : fInitialized(false) {}
    virtual ~KElectricFieldSolver() = default;

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

    double Potential(const KPosition& P) const
    {
        return PotentialCore(P);
    }

    KFieldVector ElectricField(const KPosition& P) const
    {
        return ElectricFieldCore(P);
    }

    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KPosition& P) const
    {
        return ElectricFieldAndPotentialCore(P);
    }

  private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;
    virtual void DeinitializeCore() = 0;

    virtual double PotentialCore(const KPosition& P) const = 0;
    virtual KFieldVector ElectricFieldCore(const KPosition& P) const = 0;

    virtual std::pair<KFieldVector, double> ElectricFieldAndPotentialCore(const KPosition& P) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P);
        KFieldVector field = ElectricFieldCore(P);

        return std::pair<KFieldVector, double>(field, potential);
    };

    bool fInitialized;
};

}  // namespace KEMField

#endif /* KELECTRICFIELDSOLVER_HH_ */
