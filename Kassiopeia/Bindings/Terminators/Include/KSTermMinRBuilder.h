#ifndef Kassiopeia_KSTermMinRBuilder_h_
#define Kassiopeia_KSTermMinRBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMinR.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMinR> KSTermMinRBuilder;

template<> inline bool KSTermMinRBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "r") {
        aContainer->CopyTo(fObject, &KSTermMinR::SetMinR);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
