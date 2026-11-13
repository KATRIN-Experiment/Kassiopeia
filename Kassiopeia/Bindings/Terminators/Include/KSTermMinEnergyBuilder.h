#ifndef Kassiopeia_KSTermMinEnergyBuilder_h_
#define Kassiopeia_KSTermMinEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMinEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose kinetic energy falls below a given value.
//! @example <ksterm_min_energy name="term_min_energy" energy="1.239"/>
typedef KComplexElement<KSTermMinEnergy> KSTermMinEnergyBuilder;

template<> inline bool KSTermMinEnergyBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "energy") {
        aContainer->CopyTo(fObject, &KSTermMinEnergy::SetMinEnergy);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
