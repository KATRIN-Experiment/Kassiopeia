#ifndef Kassiopeia_KSTermMagnetronBuilder_h_
#define Kassiopeia_KSTermMagnetronBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMagnetron.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMagnetron> KSTermMagnetronBuilder;

template<> inline bool KSTermMagnetronBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "max_phi") {
        aContainer->CopyTo(fObject, &KSTermMagnetron::SetMaxPhi);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
