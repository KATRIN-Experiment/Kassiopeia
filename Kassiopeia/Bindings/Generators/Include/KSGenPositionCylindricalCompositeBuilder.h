#ifndef Kassiopeia_KSGenPositionCylindricalCompositeBuilder_h_
#define Kassiopeia_KSGenPositionCylindricalCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenPositionCylindricalComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

//! creates the position of a generated particle according to the value objects supplied, interpreting them to be cylindrical coordinates.
//! r values are in meters, phi values are in degrees, and z values are in meters.
//! @example
//! <ksgen_position_cylindrical_composite name="position_cylindrical_composite" surface="world/top">
//!     <r_cylindrical radius_min="1." radius_max="2."/>
//!     <phi_set value_start="45." value_stop="90." value_count="10"/>
//!     <z_formula value_min="2." value_max="4." value_formula="x * x / 2. + 1."/>
//! </ksgen_position_cylindrical_composite>
typedef KComplexElement<KSGenPositionCylindricalComposite> KSGenPositionCylindricalCompositeBuilder;

template<> inline bool KSGenPositionCylindricalCompositeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surface") {
        KGSurface* tSurface = KGInterface::GetInstance()->RetrieveSurface(aContainer->AsString());
        fObject->SetOrigin(tSurface->GetOrigin());
        fObject->SetXAxis(tSurface->GetXAxis());
        fObject->SetYAxis(tSurface->GetYAxis());
        fObject->SetZAxis(tSurface->GetZAxis());
        return true;
    }
    if (aContainer->GetName() == "space") {
        KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace(aContainer->AsString());
        fObject->SetOrigin(tSpace->GetOrigin());
        fObject->SetXAxis(tSpace->GetXAxis());
        fObject->SetYAxis(tSpace->GetYAxis());
        fObject->SetZAxis(tSpace->GetZAxis());
        return true;
    }
    if (aContainer->GetName() == "r") {
        fObject->SetRValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "phi") {
        fObject->SetPhiValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "z") {
        fObject->SetZValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    return false;
}

template<> inline bool KSGenPositionCylindricalCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 1) == "r") {
        aContainer->ReleaseTo(fObject, &KSGenPositionCylindricalComposite::SetRValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 3) == "phi") {
        aContainer->ReleaseTo(fObject, &KSGenPositionCylindricalComposite::SetPhiValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 1) == "z") {
        aContainer->ReleaseTo(fObject, &KSGenPositionCylindricalComposite::SetZValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
