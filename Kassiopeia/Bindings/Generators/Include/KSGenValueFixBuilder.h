#ifndef Kassiopeia_KSGenValueFixBuilder_h_
#define Kassiopeia_KSGenValueFixBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueFix.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single fixed value.
//! @example <ksgen_value_fix name="value_fix" value="1.3"/>
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
