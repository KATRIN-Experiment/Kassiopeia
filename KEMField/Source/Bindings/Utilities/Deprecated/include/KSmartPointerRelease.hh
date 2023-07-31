/*
 * KSmartPointerRelease.hh
 *
 *  Created on: 3 Aug 2015
 *      Author: wolfgang
 */

#ifndef KSMARTPOINTERRELEASE_HH_
#define KSMARTPOINTERRELEASE_HH_

#include "KContainer.hh"
#include "KException.h"
#include "KSmartPointer.hh"

namespace katrin
{

template<typename Content> KEMField::KSmartPointer<Content> ReleaseToSmartPtr(KContainer* aContainer)
{
    return ReleaseToSmartPtr<Content>(*aContainer);
}

template<typename Content> KEMField::KSmartPointer<Content> ReleaseToSmartPtr(KContainer& aContainer)
{
    Content* pointer(nullptr);
    aContainer.ReleaseTo(pointer);
    if (pointer)
        return KEMField::KSmartPointer<Content>(pointer);
    throw KException() << "Release to KSmartPointer failed due to empty"
                          " KContainer or incompatible type.";
}

template<typename Content>
KEMField::KSmartPointer<Content> ReleaseToSmartPtr(const KEMField::KSmartPointer<KContainer>& container)
{
    return ReleaseToSmartPtr<Content>(&(*container));
}

}  // namespace katrin

#endif /* KSMARTPOINTERRELEASE_HH_ */
