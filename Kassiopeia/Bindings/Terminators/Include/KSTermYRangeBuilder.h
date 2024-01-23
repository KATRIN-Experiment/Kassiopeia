#ifndef Kassiopeia_KSTermYRangeBuilder_h_
#define Kassiopeia_KSTermYRangeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermYRange.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermYRange> KSTermYRangeBuilder;

template<> inline bool KSTermYRangeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "ymin") {
        aContainer->CopyTo(fObject, &KSTermYRange::SetMinY);
        return true;
    }
    if (aContainer->GetName() == "ymax") {
        aContainer->CopyTo(fObject, &KSTermYRange::SetMaxY);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
