/*
 * KCachedChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_INCLUDE_KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_INCLUDE_KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_

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

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_INCLUDE_KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_ */
