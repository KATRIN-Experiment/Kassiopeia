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
        if (KStringUtils::IContains(tok, "mol"))
            fObject->SetMode(KSGenValueAngleCosine::EDistributionMode::MolecularFlow);
        else if (KStringUtils::IContains(tok, "clas"))
            fObject->SetMode(KSGenValueAngleCosine::EDistributionMode::Classic);
        else {
            objctmsg(eError) << "ksgen_value_angle_cosine: invalid mode <" << tok << ">"
                             << "ksgen_value_angle_cosine: valid modes are <molecular_flow> or <classic>" << eom;
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
        if (KStringUtils::IContains(tok, "for"))
            fObject->SetDirection(KSGenValueAngleCosine::EDirection::Forward);
        else if (KStringUtils::IContains(tok, "back"))
            fObject->SetDirection(KSGenValueAngleCosine::EDirection::Backward);
        else {
            objctmsg(eError) << "ksgen_value_angle_cosine: invalid direction <" << tok << ">"
                             << "ksgen_value_angle_cosine: valid directions are <forward> or <backward>" << eom;
            return false;
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
