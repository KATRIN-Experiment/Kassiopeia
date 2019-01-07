/*
 * KCachedChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_
#define KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KCachedChargeDensitySolver.hh"

namespace katrin {

typedef KComplexElement< KEMField::KCachedChargeDensitySolver > KCachedChargeDensitySolverBuilder;

template< >
inline bool KCachedChargeDensitySolverBuilder::AddAttribute( KContainer* aContainer )
{
	if( aContainer->GetName() == "name" )
	{
		std::string name;
		aContainer->CopyTo( name );
		fObject->SetName( name );
		return true;
	}
	if( aContainer->GetName() == "hash" )
	{
		std::string hash;
		aContainer->CopyTo( hash );
		fObject->SetHash( hash );
		return true;
	}
	return false;
}

} // katrin

#endif /* KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_ */
