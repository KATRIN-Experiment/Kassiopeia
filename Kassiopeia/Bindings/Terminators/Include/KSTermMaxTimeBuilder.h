#ifndef Kassiopeia_KSTermMaxTimeBuilder_h_
#define Kassiopeia_KSTermMaxTimeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxTime.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose track time rises above a given value.
//! @example <ksterm_max_time name="term_max_time" time="1.239"/>
typedef KComplexElement<KSTermMaxTime> KSTermMaxTimeBuilder;

template<> inline bool KSTermMaxTimeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "time") {
        aContainer->CopyTo(fObject, &KSTermMaxTime::SetTime);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
