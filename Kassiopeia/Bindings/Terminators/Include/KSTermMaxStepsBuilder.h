#ifndef Kassiopeia_KSTermMaxStepsBuilder_h_
#define Kassiopeia_KSTermMaxStepsBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxSteps.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose track's step count rises above a given value.
//! @example <ksterm_max_steps name="term_max_steps" steps="1239"/>
typedef KComplexElement<KSTermMaxSteps> KSTermMaxStepsBuilder;

template<> inline bool KSTermMaxStepsBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "steps") {
        aContainer->CopyTo(fObject, &KSTermMaxSteps::SetMaxSteps);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
