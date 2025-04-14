#ifndef Kassiopeia_KSGenValueAngleCosineBuilder_h_
#define Kassiopeia_KSGenValueAngleCosineBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueAngleCosine.h"

#include "KStringUtils.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueAngleCosine> KSGenValueAngleCosineBuilder;

template<> inline bool KSGenValueAngleCosineBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "mode") {
        const std::string& tok = aContainer->AsReference<std::string>();
        if (KBaseStringUtils::IEquals(tok, "molecular"))
            fObject->SetMode(KSGenValueAngleCosine::EDistributionMode::MolecularFlow);
        else if (KBaseStringUtils::IEquals(tok, "classic"))
            fObject->SetMode(KSGenValueAngleCosine::EDistributionMode::Classic);
        else {
            objctmsg(eError) << "ksgen_value_angle_cosine: invalid mode <" << tok << ">" << "\n"
                             << "ksgen_value_angle_cosine: Valid modes are <molecular> or <classic>" << eom;
            return false;
        }
        return true;
    }
    if (aContainer->GetName() == "angle_min") {
        aContainer->CopyTo(fObject, &KSGenValueAngleCosine::SetAngleMin);
        return true;
    }
    if (aContainer->GetName() == "angle_max") {
        aContainer->CopyTo(fObject, &KSGenValueAngleCosine::SetAngleMax);
        return true;
    }
    if (aContainer->GetName() == "direction") {
        const std::string& tok = aContainer->AsReference<std::string>();
        if (KBaseStringUtils::IEquals(tok, "forward"))
            fObject->SetDirection(KSGenValueAngleCosine::EDirection::Forward);
        else if (KBaseStringUtils::IEquals(tok, "backward"))
            fObject->SetDirection(KSGenValueAngleCosine::EDirection::Backward);
        else {
            objctmsg(eError) << "ksgen_value_angle_cosine: invalid direction <" << tok << ">" << "\n"
                             << "ksgen_value_angle_cosine: valid directions are <forward> or <backward>" << eom;
            return false;
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
