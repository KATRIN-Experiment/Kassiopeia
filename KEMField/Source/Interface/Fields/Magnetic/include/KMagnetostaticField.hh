/*
 * KMagnetostaticField.hh
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICFIELD_HH_
#define KMAGNETOSTATICFIELD_HH_

#include "KMagneticField.hh"

namespace KEMField
{

class KMagnetostaticField : public KMagneticField
{
  public:
    KMagnetostaticField() {}
    ~KMagnetostaticField() override {}

    static std::string Name()
    {
        return "MagnetostaticField";
    }

    using KMagneticField::MagneticPotential;

    KThreeVector MagneticPotential(const KPosition& P) const
    {
        return MagneticPotentialCore(P);
    }

    KThreeVector MagneticField(const KPosition& P) const
    {
        return MagneticFieldCore(P);
    }

    KGradient MagneticGradient(const KPosition& P) const
    {
        return MagneticGradientCore(P);
    }

  private:
    KThreeVector MagneticPotentialCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticPotentialCore(P);
    }

    KThreeVector MagneticFieldCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticFieldCore(P);
    }

    KGradient MagneticGradientCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticGradientCore(P);
    }

    std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P,
                                                                    const double& /*time*/) const override
    {
        return MagneticFieldAndGradientCore(P);
    }

    virtual KThreeVector MagneticPotentialCore(const KPosition& P) const = 0;
    virtual KThreeVector MagneticFieldCore(const KPosition& P) const = 0;
    virtual KGradient MagneticGradientCore(const KPosition& P) const = 0;
    virtual std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KThreeVector field = MagneticFieldCore(P);
        KGradient grad = MagneticGradientCore(P);

        return std::pair<KThreeVector, KGradient>(field, grad);
    }
};
}  // namespace KEMField


#endif /* KMAGNETOSTATICFIELD_HH_ */
