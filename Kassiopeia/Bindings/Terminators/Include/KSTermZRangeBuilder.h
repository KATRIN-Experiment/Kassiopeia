#ifndef Kassiopeia_KSTermZRangeBuilder_h_
#define Kassiopeia_KSTermZRangeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermZRange.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermZRange> KSTermZRangeBuilder;

template<> inline bool KSTermZRangeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "zmin") {
        aContainer->CopyTo(fObject, &KSTermZRange::SetMinZ);
        return true;
    }
    if (aContainer->GetName() == "zmax") {
        aContainer->CopyTo(fObject, &KSTermZRange::SetMaxZ);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
