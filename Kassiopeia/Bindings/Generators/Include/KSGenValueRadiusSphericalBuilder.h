#ifndef Kassiopeia_KSGenValueRadiusSphericalBuilder_h_
#define Kassiopeia_KSGenValueRadiusSphericalBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueRadiusSpherical.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueRadiusSpherical> KSGenValueRadiusSphericalBuilder;

template<> inline bool KSGenValueRadiusSphericalBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "radius_min") {
        aContainer->CopyTo(fObject, &KSGenValueRadiusSpherical::SetRadiusMin);
        return true;
    }
    if (aContainer->GetName() == "radius_max") {
        aContainer->CopyTo(fObject, &KSGenValueRadiusSpherical::SetRadiusMax);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
