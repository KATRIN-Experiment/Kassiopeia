#ifndef Kassiopeia_KSRootStepModifierBuilder_h_
#define Kassiopeia_KSRootStepModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootStepModifier.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSRootStepModifier> KSRootStepModifierBuilder;

template<> inline bool KSRootStepModifierBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "add_modifier") {
        fObject->AddModifier(KToolbox::GetInstance().Get<KSStepModifier>(aContainer->AsString()));
        return true;
    }
    return false;
}
}  // namespace katrin

#endif  //Kassiopeia_KSRootStepModifierBuilder_h_
