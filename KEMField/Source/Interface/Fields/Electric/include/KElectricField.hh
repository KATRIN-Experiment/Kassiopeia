/*
 * KElectricField.hh
 *
 *  Created on: 11.05.2015
 *      Author: gosda
 */

#ifndef KELECTRICFIELD_HH_
#define KELECTRICFIELD_HH_

#include "KNamed.h"
#include "KThreeVector_KEMField.hh"

#include <string>

namespace KEMField
{
class KElectricField : public katrin::KNamed
{
  public:
    KElectricField() : katrin::KNamed(), fInitialized(false) {}
    ~KElectricField() override = default;

    // this class uses the non virtual interface (NVI) pattern
    // the virtual methods that have to be implemented by subclasses
    // are named ...Core

    double Potential(const KPosition& P, const double& time) const
    {
        return PotentialCore(P, time);
    }

    KFieldVector ElectricField(const KPosition& P, const double& time) const
    {
        return ElectricFieldCore(P, time);
    }

    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KPosition& P, const double& time) const
    {
        return ElectricFieldAndPotentialCore(P, time);
    }


    void Initialize()
    {
        if (!fInitialized) {
            InitializeCore();
            fInitialized = true;
        }
    }

  private:
    virtual double PotentialCore(const KPosition& P, const double& time) const = 0;

    virtual KFieldVector ElectricFieldCore(const KPosition& P, const double& time) const = 0;

    virtual std::pair<KFieldVector, double> ElectricFieldAndPotentialCore(const KPosition& P, const double& time) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P, time);
        KFieldVector field = ElectricFieldCore(P, time);

        return std::pair<KFieldVector, double>(field, potential);
    }

    virtual void InitializeCore() {}

    bool fInitialized;

    std::string fName;
};

}  // namespace KEMField


#endif /* KELECTRICFIELD_HH_ */
