#ifndef Kassiopeia_KSGenValueRadiusCylindricalBuilder_h_
#define Kassiopeia_KSGenValueRadiusCylindricalBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueRadiusCylindrical.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single random value drawn from a distribution suitable for randomly distributing radii in cylindrical coordinates.
//! @example <ksgen_value_radius_cylindrical name="value_radius_cylindrical" radius_min="5." radius_max="10."/>
typedef KComplexElement<KSGenValueRadiusCylindrical> KSGenValueRadiusCylindricalBuilder;

template<> inline bool KSGenValueRadiusCylindricalBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "radius_min") {
        aContainer->CopyTo(fObject, &KSGenValueRadiusCylindrical::SetRadiusMin);
        return true;
    }
    if (aContainer->GetName() == "radius_max") {
        aContainer->CopyTo(fObject, &KSGenValueRadiusCylindrical::SetRadiusMax);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
