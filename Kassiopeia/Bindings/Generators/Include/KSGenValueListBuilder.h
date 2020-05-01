#ifndef Kassiopeia_KSGenValueListBuilder_h_
#define Kassiopeia_KSGenValueListBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueList.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueList> KSGenValueListBuilder;

template<> inline bool KSGenValueListBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "add_value") {
        aContainer->CopyTo(fObject, &KSGenValueList::AddValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
