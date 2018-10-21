/*
 * KChargeDensitySolver.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_KELECTRICFIELDSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_KELECTRICFIELDSOLVER_HH_

#include "KSurfaceContainer.hh"
#include "KEMThreeVector.hh"

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

    KEMThreeVector ElectricField(const KPosition& P) const {
        return ElectricFieldCore(P);
    }

    std::pair<KEMThreeVector,double> ElectricFieldAndPotential(const KPosition& P) const
    {
        return ElectricFieldAndPotentialCore(P);
    }

private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;
    virtual double PotentialCore(const KPosition& P ) const = 0;
    virtual KEMThreeVector ElectricFieldCore( const KPosition& P) const = 0;

    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P);
        KEMThreeVector field = ElectricFieldCore(P);

        return std::pair<KEMThreeVector,double>(field,potential);
    };

    bool fInitialized;
};

}

#endif /* KEMFIELD_SOURCE_2_0_FIELDSOLVERS_ELECTRIC_KELECTRICFIELDSOLVER_HH_ */
