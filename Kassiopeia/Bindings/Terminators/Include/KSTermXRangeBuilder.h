#ifndef Kassiopeia_KSTermXRangeBuilder_h_
#define Kassiopeia_KSTermXRangeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermXRange.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermXRange> KSTermXRangeBuilder;

template<> inline bool KSTermXRangeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "xmin") {
        aContainer->CopyTo(fObject, &KSTermXRange::SetMinX);
        return true;
    }
    if (aContainer->GetName() == "xmax") {
        aContainer->CopyTo(fObject, &KSTermXRange::SetMaxX);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
