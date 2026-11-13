#ifndef Kassiopeia_KSTermMaxEnergyBuilder_h_
#define Kassiopeia_KSTermMaxEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose kinetic energy rises above a given value.
//! @example <ksterm_max_energy name="term_max_energy" energy="1.239"/>
typedef KComplexElement<KSTermMaxEnergy> KSTermMaxEnergyBuilder;

template<> inline bool KSTermMaxEnergyBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "energy") {
        aContainer->CopyTo(fObject, &KSTermMaxEnergy::SetMaxEnergy);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
