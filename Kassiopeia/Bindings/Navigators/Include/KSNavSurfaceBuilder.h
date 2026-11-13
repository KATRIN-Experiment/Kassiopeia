#ifndef Kassiopeia_KSNavSurfaceBuilder_h_
#define Kassiopeia_KSNavSurfaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavSurface.h"

using namespace Kassiopeia;
namespace katrin
{

//! this is the principal surface navigator within kassiopeia.
//! given a particle state, it determines whether the particle has been transmitted through the surface or reflected from it, and what space it subsequently enters.
//! @example <ksnav_surface name="nav_surface" transmission_split="false" reflection_split="false"/>
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
