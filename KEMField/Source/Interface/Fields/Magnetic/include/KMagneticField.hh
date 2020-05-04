/*
 * KMagneticField.hh
 *
 *  Created on: 20.05.2015
 *      Author: gosda
 */

#ifndef KMAGNETICFIELD_HH_
#define KMAGNETICFIELD_HH_

#include "KNamed.h"
#include "KThreeMatrix_KEMField.hh"
#include "KThreeVector_KEMField.hh"

namespace KEMField
{

class KMagneticField : public katrin::KNamed
{
  public:
    KMagneticField() : katrin::KNamed(), fInitialized(false) {}
    virtual ~KMagneticField() {}

    // this class uses the non virtual interface (NVI) pattern
    // the virtual methods that have to be implemented by subclasses
    // are named ...Core

    KThreeVector MagneticPotential(const KPosition& P, const double& time) const
    {
        return MagneticPotentialCore(P, time);
    }

    KThreeVector MagneticField(const KPosition& P, const double& time) const
    {
        return MagneticFieldCore(P, time);
    }

    KGradient MagneticGradient(const KPosition& P, const double& time) const
    {
        return MagneticGradientCore(P, time);
    }

    std::pair<KThreeVector, KGradient> MagneticFieldAndGradient(const KPosition& P, const double& time) const
    {
        return MagneticFieldAndGradientCore(P, time);
    }


    void Initialize()
    {
        if (!fInitialized) {
            InitializeCore();
            fInitialized = true;
        }
    }

  protected:
    virtual KThreeVector MagneticPotentialCore(const KPosition& P, const double& time) const = 0;

    virtual KThreeVector MagneticFieldCore(const KPosition& P, const double& time) const = 0;

    virtual KGradient MagneticGradientCore(const KPosition& P, const double& time) const = 0;

    virtual std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P,
                                                                            const double& time) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KThreeVector field = MagneticFieldCore(P, time);
        KGradient grad = MagneticGradientCore(P, time);

        return std::pair<KThreeVector, KGradient>(field, grad);
    }

    virtual void InitializeCore() {}

    bool fInitialized;

    std::string fName;
};

}  // namespace KEMField


#endif /* KMAGNETICFIELD_HH_ */
