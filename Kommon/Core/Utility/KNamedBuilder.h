//
// Created by trost on 26.07.16.
//
#include "KComplexElement.hh"
#include "KTagged.h"

#ifndef KASPER_KNAMEDBUILDER_H
#define KASPER_KNAMEDBUILDER_H

namespace katrin
{

class KNamedReference : public KNamed
{};

typedef KComplexElement<KNamedReference> KNamedBuilder;

template<> inline bool KNamedBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "Name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }

    return false;
}

}  // namespace katrin
#endif  //KASPER_KNAMEDBUILDER_H
