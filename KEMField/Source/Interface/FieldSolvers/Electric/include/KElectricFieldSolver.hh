/*
 * KChargeDensitySolver.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KELECTRICFIELDSOLVER_HH_
#define KELECTRICFIELDSOLVER_HH_

#include "KThreeVector_KEMField.hh"
#include "KSurfaceContainer.hh"

namespace KEMField {

class KElectricFieldSolver
{
public:
    KElectricFieldSolver() :fInitialized(false) {}
    virtual ~KElectricFieldSolver() {}

    void Initialize( KSurfaceContainer& container) {
        if(!fInitialized) {
            InitializeCore(container);
            fInitialized = true;
        }
    }

    double Potential(const KPosition& P) const {
        return PotentialCore(P);
    }

    KThreeVector ElectricField(const KPosition& P) const {
        return ElectricFieldCore(P);
    }

    std::pair<KThreeVector,double> ElectricFieldAndPotential(const KPosition& P) const
    {
        return ElectricFieldAndPotentialCore(P);
    }

private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;
    virtual double PotentialCore(const KPosition& P ) const = 0;
    virtual KThreeVector ElectricFieldCore( const KPosition& P) const = 0;

    virtual std::pair<KThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P);
        KThreeVector field = ElectricFieldCore(P);

        return std::pair<KThreeVector,double>(field,potential);
    };

    bool fInitialized;
};

}

#endif /* KELECTRICFIELDSOLVER_HH_ */
