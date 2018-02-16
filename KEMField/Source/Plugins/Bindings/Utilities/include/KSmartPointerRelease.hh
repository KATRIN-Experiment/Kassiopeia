/*
 * KSmartPointerRelease.hh
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_TOOLBOX_KSMARTPOINTERRELEASE_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_TOOLBOX_KSMARTPOINTERRELEASE_HH_

#include "KSmartPointer.hh"
#include "KContainer.hh"
#include "KException.h"

namespace katrin {

template< typename Content >
KEMField::KSmartPointer<Content> ReleaseToSmartPtr(KContainer* aContainer){
	return ReleaseToSmartPtr<Content>(*aContainer);
}

template< typename Content >
KEMField::KSmartPointer<Content> ReleaseToSmartPtr(KContainer& aContainer)
{
	Content* pointer(NULL);
	aContainer.ReleaseTo(pointer);
	if(pointer)
		return KEMField::KSmartPointer<Content>(pointer);
	throw KException() << "Release to KSmartPointer failed due to empty"
			" KContainer or incompatible type.";
}

template< typename Content >
KEMField::KSmartPointer<Content> ReleaseToSmartPtr(
		KEMField::KSmartPointer<KContainer> container)
{
	return ReleaseToSmartPtr<Content>(&(*container));
}

} // katrin

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_TOOLBOX_KSMARTPOINTERRELEASE_HH_ */
