/*
 * KMagnetostaticField.hh
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICFIELD_HH_

#include "KMagneticField.hh"

namespace KEMField
{

class KMagnetostaticField : public KMagneticField
{
public:
	KMagnetostaticField() {}
	virtual ~KMagnetostaticField() {}

	static std::string Name() {return "MagnetostaticField";}

	using KMagneticField::MagneticPotential;

	KEMThreeVector MagneticPotential(const KPosition& P) const {
		return MagneticPotentialCore(P);
	}

	KEMThreeVector MagneticField(const KPosition& P) const {
	    return MagneticFieldCore(P);
	}

	KGradient MagneticGradient(const KPosition& P) const {
	    return MagneticGradientCore(P);
	}

private:

	virtual KEMThreeVector MagneticPotentialCore(const KPosition& P, const double& /*time*/) const
	{
		return MagneticPotentialCore(P);
	}

	virtual KEMThreeVector MagneticFieldCore(const KPosition& P, const double& /*time*/) const
	{
		return MagneticFieldCore(P);
	}

	virtual KGradient MagneticGradientCore(const KPosition& P, const double& /*time*/) const
	{
		return MagneticGradientCore(P);
	}

    virtual std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P, const double& /*time*/) const
    {
        return MagneticFieldAndGradientCore(P);
    }

	virtual KEMThreeVector MagneticPotentialCore(const KPosition& P) const = 0;
	virtual KEMThreeVector MagneticFieldCore(const KPosition& P) const = 0;
	virtual KGradient MagneticGradientCore(const KPosition& P) const = 0;
    virtual std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KEMThreeVector field = MagneticFieldCore(P);
        KGradient grad = MagneticGradientCore(P);

        return std::pair<KEMThreeVector, KGradient>(field,grad);
    }

};
} // KEMField



#endif /* KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICFIELD_HH_ */
