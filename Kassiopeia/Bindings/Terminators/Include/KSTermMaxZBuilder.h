#ifndef Kassiopeia_KSTermMaxZBuilder_h_
#define Kassiopeia_KSTermMaxZBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxZ.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose position's z coordinate rises above a given value.
//! @example <ksterm_max_z name="term_max_z" z="1.239"/>
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
