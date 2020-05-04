#ifndef Kassiopeia_KSGenValueAngleSphericalBuilder_h_
#define Kassiopeia_KSGenValueAngleSphericalBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueAngleSpherical.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueAngleSpherical> KSGenValueAngleSphericalBuilder;

template<> inline bool KSGenValueAngleSphericalBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "angle_min") {
        aContainer->CopyTo(fObject, &KSGenValueAngleSpherical::SetAngleMin);
        return true;
    }
    if (aContainer->GetName() == "angle_max") {
        aContainer->CopyTo(fObject, &KSGenValueAngleSpherical::SetAngleMax);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
