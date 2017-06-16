/*
 * KMagneticField.hh
 *
 *  Created on: 20.05.2015
 *      Author: gosda
 */

#ifndef KMAGNETICFIELD_HH_
#define KMAGNETICFIELD_HH_

#include "KEMThreeVector.hh"
#include "KEMThreeMatrix.hh"

namespace KEMField
{

class KMagneticField
{
public:
	KMagneticField() : fInitialized( false ){}
	virtual ~KMagneticField() {}

	std::string Name() {return fName; }
    // this class uses the non virtual interface (NVI) pattern
    // the virtual methods that have to be implemented by subclasses
    // are named ...Core

	KEMThreeVector MagneticPotential(const KPosition& P, const double& time) const {
		return MagneticPotentialCore(P,time);
	}

	KEMThreeVector MagneticField(const KPosition& P, const double& time) const {
		return MagneticFieldCore(P,time);
	}

	KGradient MagneticGradient(const KPosition& P, const double& time) const {
		return MagneticGradientCore(P,time);
	}

    std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradient(const KPosition& P, const double& time) const
    {
        return MagneticFieldAndGradientCore(P,time);
    }


	void Initialize() {
	    if(!fInitialized)
	    {
	        InitializeCore();
	        fInitialized = true;
	    }
	}

    void SetName(std::string name){fName = name;}

protected:

	virtual KEMThreeVector MagneticPotentialCore(
			const KPosition& P, const double& time) const = 0;

	virtual KEMThreeVector MagneticFieldCore(
			const KPosition& P, const double& time) const = 0;

	virtual KGradient MagneticGradientCore(
			const KPosition& P, const double& time) const = 0;

    virtual std::pair<KEMThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P, const double& time) const
    {
        //default behavior is to simply call the field and gradient separately
        //this function may be overloaded to perform a more efficient combined calculation
        KEMThreeVector field = MagneticFieldCore(P,time);
        KGradient grad = MagneticGradientCore(P,time);

        return std::pair<KEMThreeVector, KGradient>(field,grad);
    }

	virtual void InitializeCore() {}

	bool fInitialized;

	std::string fName;

};

} //KEMField



#endif /* KMAGNETICFIELD_HH_ */
