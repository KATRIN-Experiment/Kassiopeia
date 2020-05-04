#ifndef Kassiopeia_KSTermMaxRBuilder_h_
#define Kassiopeia_KSTermMaxRBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxR.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMaxR> KSTermMaxRBuilder;

template<> inline bool KSTermMaxRBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "r") {
        aContainer->CopyTo(fObject, &KSTermMaxR::SetMaxR);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
