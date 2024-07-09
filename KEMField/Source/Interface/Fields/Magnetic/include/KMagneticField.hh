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
    ~KMagneticField() override = default;

    // this class uses the non virtual interface (NVI) pattern
    // the virtual methods that have to be implemented by subclasses
    // are named ...Core

    KFieldVector MagneticPotential(const KPosition& P, const double& time) const
    {
        return MagneticPotentialCore(P, time);
    }

    KFieldVector MagneticField(const KPosition& P, const double& time) const
    {
        return MagneticFieldCore(P, time);
    }

    KGradient MagneticGradient(const KPosition& P, const double& time) const
    {
        return MagneticGradientCore(P, time);
    }

    std::pair<KFieldVector, KGradient> MagneticFieldAndGradient(const KPosition& P, const double& time) const
    {
        return MagneticFieldAndGradientCore(P, time);
    }

    bool Check(const KPosition& P, const double& time) const
    {
        return CheckCore(P, time);
    }

    void Initialize()
    {
        if (!fInitialized) {
            InitializeCore();
            fInitialized = true;
        }
    }

    void Deinitialize()
    {
        if (fInitialized) {
            DeinitializeCore();
        }
        fInitialized = false;
    }

  protected:
    virtual KFieldVector MagneticPotentialCore(const KPosition& P, const double& time) const = 0;

    virtual KFieldVector MagneticFieldCore(const KPosition& P, const double& time) const = 0;

    virtual KGradient MagneticGradientCore(const KPosition& P, const double& time) const = 0;

    virtual std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P,
                                                                            const double& time) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KFieldVector field = MagneticFieldCore(P, time);
        KGradient grad = MagneticGradientCore(P, time);

        return std::pair<KFieldVector, KGradient>(field, grad);
    }

    virtual bool CheckCore(const KPosition& /*P*/, const double& /*time*/) const
    {
        //default behavior is to assume all points are valid
        return true;
    }

    virtual void InitializeCore() {}
    virtual void DeinitializeCore() {}

    bool fInitialized;

    std::string fName;
};

}  // namespace KEMField


#endif /* KMAGNETICFIELD_HH_ */
