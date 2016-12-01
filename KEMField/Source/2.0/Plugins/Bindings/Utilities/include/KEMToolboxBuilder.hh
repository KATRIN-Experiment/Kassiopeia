/*
 * KEMToolboxBuilder.hh
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */

#ifndef KEMTOOLBOXBUILDER_HH_
#define KEMTOOLBOXBUILDER_HH_

#include "KComplexElement.hh"
#include <iostream>
#include "KToolbox.h"

namespace katrin
{
	class KEMRoot{};
	typedef KComplexElement< KEMRoot > KEMToolboxBuilder;

	template< >
	inline bool KEMToolboxBuilder::AddElement( KContainer* aContainer)
	{
	    if( !aContainer->Empty() ) {
	        KToolbox::GetInstance().AddContainer(*aContainer,aContainer->GetName());
	        return true;
	    }
	    else return false;
	}
} //katrin



#endif /* KEMTOOLBOXBUILDER_HH_ */
