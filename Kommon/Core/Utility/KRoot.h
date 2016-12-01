//
// Created by trost on 12.07.16.
//

#ifndef KASPER_KROOT_H
#define KASPER_KROOT_H


#include "KComplexElement.hh"
#include "KApplicationRunner.h"
#include "KToolbox.h"


namespace katrin
{


typedef KComplexElement<katrin::KToolbox> KRootBuilder;

template<>
inline bool KRootBuilder::Begin()
{
    return true;
}

template<>
inline bool KRootBuilder::AddElement(KContainer *aContainer)
{
    if (aContainer->Empty()) {
        return true;
    }
    if (aContainer->Is<KApplicationRunner>()) {
        return aContainer->AsPointer<KApplicationRunner>()->Execute();
    }
    if( aContainer->Is< KTagged >() )
    {
        KToolbox::GetInstance().AddContainer(*aContainer);
        return true;
    }


    return false;
}

template<>
inline bool KRootBuilder::End()
{
    return true;
}

}

#endif //KASPER_KROOT_H
