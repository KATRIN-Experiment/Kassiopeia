#ifndef Kassiopeia_KSGenValueUniformBuilder_h_
#define Kassiopeia_KSGenValueUniformBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueUniform.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single random value drawn from a portion of a uniform distribution.
//! @example <ksgen_value_uniform name="value_uniform" value_min="-10." value_max="10."/>
typedef KComplexElement<KSGenValueUniform> KSGenValueUniformBuilder;

template<> inline bool KSGenValueUniformBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value_min") {
        aContainer->CopyTo(fObject, &KSGenValueUniform::SetValueMin);
        return true;
    }
    if (aContainer->GetName() == "value_max") {
        aContainer->CopyTo(fObject, &KSGenValueUniform::SetValueMax);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
