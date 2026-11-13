#ifndef Kassiopeia_KSGenEnergyCompositeBuilder_h_
#define Kassiopeia_KSGenEnergyCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenEnergyComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

//! creates the kinetic energy of a generated particle according to the value objects supplied.
//! energy is in electron volts.
//! @example
//! <ksgen_energy_composite name="energy_composite">
//!     <energy_gauss value_min="18500." value_max="18700." value_mean="18600." value_sigma="50."/>
//! </ksgen_energy_composite>
typedef KComplexElement<KSGenEnergyComposite> KSGenEnergyCompositeBuilder;

template<> inline bool KSGenEnergyCompositeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "energy") {
        fObject->SetEnergyValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    return false;
}

template<> inline bool KSGenEnergyCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 6) == "energy") {
        aContainer->ReleaseTo(fObject, &KSGenEnergyComposite::SetEnergyValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
