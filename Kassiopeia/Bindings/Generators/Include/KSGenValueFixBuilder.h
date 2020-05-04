#ifndef Kassiopeia_KSGenValueFixBuilder_h_
#define Kassiopeia_KSGenValueFixBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueFix.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueFix> KSGenValueFixBuilder;

template<> inline bool KSGenValueFixBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value") {
        aContainer->CopyTo(fObject, &KSGenValueFix::SetValue);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
