/*
 * KElectricQuadrupoleField.hh
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELD_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELD_HH_

#include "KElectrostaticField.hh"

namespace KEMField {

class KElectricQuadrupoleField: public KElectrostaticField {
public:
	KElectricQuadrupoleField();
	virtual ~KElectricQuadrupoleField();

	void SetLocation( const KPosition& aLocation );
	void SetStrength( const double& aStrength );
	void SetLength( const double& aLength );
	void SetRadius( const double& aRadius );

private:
	double PotentialCore( const KPosition& aSamplePoint) const;
	KEMThreeVector ElectricFieldCore( const KPosition& aSamplePoint) const;

	KPosition fLocation;
	double fStrength;
	double fLength;
	double fRadius;
	double fCharacteristic;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELD_HH_ */
