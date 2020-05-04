#ifndef Kassiopeia_KSRootEventModifierBuilder_h_
#define Kassiopeia_KSRootEventModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootEventModifier.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSRootEventModifier> KSRootEventModifierBuilder;

template<> inline bool KSRootEventModifierBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "add_modifier") {
        fObject->AddModifier(KToolbox::GetInstance().Get<KSEventModifier>(aContainer->AsReference<std::string>()));
        return true;
    }
    return false;
}
}  // namespace katrin

#endif  //Kassiopeia_KSRootEventModifierBuilder_h_
