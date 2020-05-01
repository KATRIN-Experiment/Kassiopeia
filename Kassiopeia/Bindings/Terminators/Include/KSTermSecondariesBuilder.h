#ifndef Kassiopeia_KSTermSecondariesBuilder_h_
#define Kassiopeia_KSTermSecondariesBuilder_h_

#include "KComplexElement.hh"
#include "KSTermSecondaries.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermSecondaries> KSTermSecondariesBuilder;

template<> inline bool KSTermSecondariesBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
