#ifndef Kassiopeia_KSNavSpaceBuilder_h_
#define Kassiopeia_KSNavSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavSpace.h"
#include "KSNavigatorsMessage.h"


using namespace Kassiopeia;
namespace katrin
{

//! this is the principal space navigator within kassiopeia.
//! given a trajectory segment, it determines whether or not tracks enter or exit spaces or cross surfaces.
//! @example <ksnav_space name="nav_space" enter_split="true" exit_split="true" tolerance="1.e-6"/>
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
