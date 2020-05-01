#ifndef Kassiopeia_KSNavSpaceBuilder_h_
#define Kassiopeia_KSNavSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavSpace.h"
#include "KSNavigatorsMessage.h"


using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSNavSpace> KSNavSpaceBuilder;

template<> inline bool KSNavSpaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "enter_split") {
        aContainer->CopyTo(fObject, &KSNavSpace::SetEnterSplit);
        return true;
    }
    if (aContainer->GetName() == "exit_split") {
        aContainer->CopyTo(fObject, &KSNavSpace::SetExitSplit);
        return true;
    }
    if (aContainer->GetName() == "fail_check") {
        aContainer->CopyTo(fObject, &KSNavSpace::SetFailCheck);
        return true;
    }
    if (aContainer->GetName() == "tolerance") {
        navmsg(eWarning)
            << "backward compatibility warning: the tolerance attribute is no longer needed in the space navigator, please remove it from your config file!"
            << eom;
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
