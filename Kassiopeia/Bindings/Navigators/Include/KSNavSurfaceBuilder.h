#ifndef Kassiopeia_KSNavSurfaceBuilder_h_
#define Kassiopeia_KSNavSurfaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavSurface.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSNavSurface> KSNavSurfaceBuilder;

template<> inline bool KSNavSurfaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "transmission_split") {
        aContainer->CopyTo(fObject, &KSNavSurface::SetTransmissionSplit);
        return true;
    }
    if (aContainer->GetName() == "reflection_split") {
        aContainer->CopyTo(fObject, &KSNavSurface::SetReflectionSplit);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
