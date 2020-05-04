#ifndef Kassiopeia_KSTermMaxZBuilder_h_
#define Kassiopeia_KSTermMaxZBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxZ.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMaxZ> KSTermMaxZBuilder;

template<> inline bool KSTermMaxZBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "z") {
        aContainer->CopyTo(fObject, &KSTermMaxZ::SetMaxZ);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
