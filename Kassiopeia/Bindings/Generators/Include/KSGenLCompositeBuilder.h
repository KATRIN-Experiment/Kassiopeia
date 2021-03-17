#ifndef Kassiopeia_KSGenLCompositeBuilder_h_
#define Kassiopeia_KSGenLCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenLComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenLComposite> KSGenLCompositeBuilder;

template<> inline bool KSGenLCompositeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "l_value") {
        fObject->SetLValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    return false;
}

template<> inline bool KSGenLCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 1) == "l") {
        aContainer->ReleaseTo(fObject, &KSGenLComposite::SetLValue);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
