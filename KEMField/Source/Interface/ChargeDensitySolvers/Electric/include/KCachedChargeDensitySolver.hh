/*
 * KCachedChargeDensitySolver.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 *
 *      Imported from KSFieldElectrostatic
 */

#ifndef KCACHEDCHARGEDENSITYSOLVER_HH_
#define KCACHEDCHARGEDENSITYSOLVER_HH_

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

#endif /* KCACHEDCHARGEDENSITYSOLVER_HH_ */
