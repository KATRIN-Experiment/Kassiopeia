/*
 * KElectricField.hh
 *
 *  Created on: 11.05.2015
 *      Author: gosda
 */

#ifndef KELECTRICFIELD_HH_
#define KELECTRICFIELD_HH_

#include <string>

#include "KThreeVector_KEMField.hh"

namespace KEMField
{
class KElectricField
{
public:
    KElectricField() : fInitialized(false) {}
    virtual ~KElectricField() {}

    std::string Name() { return fName; }

    // this class uses the non virtual interface (NVI) pattern
    // the virtual methods that have to be implemented by subclasses
    // are named ...Core

    double Potential(const KPosition& P, const double& time) const {
        return PotentialCore(P,time);
    }

    KThreeVector ElectricField(const KPosition& P, const double& time) const {
        return ElectricFieldCore(P,time);
    }

    std::pair<KThreeVector,double> ElectricFieldAndPotential(const KPosition& P, const double& time) const
    {
        return ElectricFieldAndPotentialCore(P,time);
    }


    void Initialize() {
        if(!fInitialized) {
            InitializeCore();
            fInitialized = true;
        }
    }

    void SetName(std::string name){fName = name;}

private:

    virtual double PotentialCore( const KPosition& P,const double& time) const = 0;

    virtual KThreeVector ElectricFieldCore(
            const KPosition& P,const double& time) const = 0;

    virtual std::pair<KThreeVector,double> ElectricFieldAndPotentialCore(const KPosition& P, const double& time) const
    {
        //the default behavior is just to call the field and potential separately

        //this routine can be overloaded to allow for additional efficiency in for some specific
        //field calculations methods which can produce the field and potential values
        //at the same time with minimal additional work (e.g. ZH and fast multipole).

        double potential = PotentialCore(P,time);
        KThreeVector field = ElectricFieldCore(P, time);

        return std::pair<KThreeVector,double>(field,potential);
    }

    virtual void InitializeCore() {}

    bool fInitialized;

    std::string fName;
};

}




#endif /* KELECTRICFIELD_HH_ */
