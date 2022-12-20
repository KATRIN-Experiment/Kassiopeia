#ifndef Kassiopeia_KSRootBuilder_h_
#define Kassiopeia_KSRootBuilder_h_

#include "KComplexElement.hh"
#include "KElectricField.hh"
#include "KMagneticField.hh"
#include "KSElectricKEMField.h"
#include "KSMagneticKEMField.h"
#include "KSMainMessage.h"
#include "KSRoot.h"
#include "KSSimulation.h"
#include "KToolbox.h"
#include "KRandom.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSRoot> KSRootBuilder;

template<> inline bool KSRootBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "random_seed") {
        const auto& seed = aContainer->AsReference<unsigned int>();
        KRandom::GetInstance().SetSeed(seed);
        return true;
    }
    return false;
}

template<> inline bool KSRootBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSSimulation>()) {
        aContainer->ReleaseTo(fObject, &KSRoot::Execute);
        return true;
    }
    if (aContainer->Is<KSObject>()) {
        KToolbox::GetInstance().AddContainer(*aContainer);
        return true;
    }

    /// NOTE: deprecated legacy support for old field bindings in the <kassiopeia> tag
    if (aContainer->Is<KEMField::KElectricField>()) {
        mainmsg(eInfo) << "legacy binding for electric field <" << aContainer->GetName()
                          << "> is DEPRECATED - please move objects to <kemfield> tag" << eom;

        auto* tField = new KSElectricKEMField();
        tField->SetName(aContainer->GetName());
        aContainer->ReleaseTo(tField, &KSElectricKEMField::SetElectricField);
        KToolbox::GetInstance().Add(tField, tField->GetName());
        return true;
    }
    if (aContainer->Is<KEMField::KMagneticField>()) {
        mainmsg(eInfo) << "legacy binding for magnetic field <" << aContainer->GetName()
                          << "> is DEPRECATED - please move objects to <kemfield> tag" << eom;

        auto* tField = new KSMagneticKEMField();
        tField->SetName(aContainer->GetName());
        aContainer->ReleaseTo(tField, &KSMagneticKEMField::SetMagneticField);
        KToolbox::GetInstance().Add(tField, tField->GetName());
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
