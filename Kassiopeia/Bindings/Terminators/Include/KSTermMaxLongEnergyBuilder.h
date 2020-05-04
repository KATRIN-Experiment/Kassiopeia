#ifndef Kassiopeia_KSTermMaxLongEnergyBuilder_h_
#define Kassiopeia_KSTermMaxLongEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxLongEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMaxLongEnergy> KSTermMaxLongEnergyBuilder;

template<> inline bool KSTermMaxLongEnergyBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "long_energy") {
        aContainer->CopyTo(fObject, &KSTermMaxLongEnergy::SetMaxLongEnergy);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
