/*
 * KCachedChargeDensitySolver.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 *
 *      Imported from KSFieldElectrostatic
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KCACHEDCHARGEDENSITYSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KCACHEDCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"

namespace KEMField {

class KCachedChargeDensitySolver :
		public KChargeDensitySolver
{
public:
	KCachedChargeDensitySolver();
	virtual ~KCachedChargeDensitySolver();

	void SetName( std::string s )
	{
		fName = s;
	}
	void SetHash( std::string s )
	{
		fHash = s;
	}

private:
    virtual void InitializeCore( KSurfaceContainer& container );

	std::string fName;
	std::string fHash;
};

} // KEMField

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KCACHEDCHARGEDENSITYSOLVER_HH_ */
