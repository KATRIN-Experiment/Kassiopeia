#ifndef Kassiopeia_KSGenValueAngleSphericalBuilder_h_
#define Kassiopeia_KSGenValueAngleSphericalBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueAngleSpherical.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single random value drawn from a distribution suitable for randomly distributing polar angles in cylindrical coordinates.
//! @example <ksgen_value_angle_spherical name="value_angle_spherical" angle_min="10." angle_max="70."/>
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
