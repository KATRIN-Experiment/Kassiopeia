#ifndef Kassiopeia_KSRootSurfaceNavigatorBuilder_h_
#define Kassiopeia_KSRootSurfaceNavigatorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootSurfaceNavigator.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSRootSurfaceNavigator> KSRootSurfaceNavigatorBuilder;

template<> inline bool KSRootSurfaceNavigatorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "set_surface_navigator") {
        fObject->SetSurfaceNavigator(KToolbox::GetInstance().Get<KSSurfaceNavigator>(aContainer->AsString()));
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
