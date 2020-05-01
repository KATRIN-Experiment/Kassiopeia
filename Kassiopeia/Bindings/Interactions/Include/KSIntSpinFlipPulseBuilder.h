#ifndef Kassiopeia_KSIntSpinFlipPulseBuilder_h_
#define Kassiopeia_KSIntSpinFlipPulseBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSpinFlipPulse.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntSpinFlipPulse> KSIntSpinFlipPulseBuilder;

template<> inline bool KSIntSpinFlipPulseBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "time") {
        aContainer->CopyTo(fObject, &KSIntSpinFlipPulse::SetTime);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
