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
    KMagnetostaticField() = default;
    ~KMagnetostaticField() override = default;

    static std::string Name()
    {
        return "MagnetostaticField";
    }

    using KMagneticField::MagneticPotential;

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

  private:
    KFieldVector MagneticPotentialCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticPotentialCore(P);
    }

    KFieldVector MagneticFieldCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticFieldCore(P);
    }

    KGradient MagneticGradientCore(const KPosition& P, const double& /*time*/) const override
    {
        return MagneticGradientCore(P);
    }

    std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P,
                                                                    const double& /*time*/) const override
    {
        return MagneticFieldAndGradientCore(P);
    }

    virtual KFieldVector MagneticPotentialCore(const KPosition& P) const = 0;
    virtual KFieldVector MagneticFieldCore(const KPosition& P) const = 0;
    virtual KGradient MagneticGradientCore(const KPosition& P) const = 0;
    virtual std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KFieldVector field = MagneticFieldCore(P);
        KGradient grad = MagneticGradientCore(P);

        return std::pair<KFieldVector, KGradient>(field, grad);
    }
};
}  // namespace KEMField


#endif /* KMAGNETOSTATICFIELD_HH_ */
