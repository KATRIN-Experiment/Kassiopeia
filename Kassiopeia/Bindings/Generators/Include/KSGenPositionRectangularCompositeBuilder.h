#ifndef Kassiopeia_KSGenPositionRectangularCompositeBuilder_h_
#define Kassiopeia_KSGenPositionRectangularCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenPositionRectangularComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

//! creates the position of a generated particle according to the value objects supplied, interpreting them in rectangular coordinates.
//! x, y, and z values are in meters.
//! @example
//! <ksgen_position_rectangular_composite name="position_rectangular_composite" space="world">
//!     <x_fix value="0."/>
//!     <y_gauss value_min="0." value_max="5." value_mean="2." value_sigma=".5"/>
//!     <z_formula value_min="2." value_max="4." value_formula="x * x / 2. + 1."/>
//! </ksgen_position_rectangular_composite>
typedef KComplexElement<KSGenPositionRectangularComposite> KSGenPositionRectangularCompositeBuilder;

template<> inline bool KSGenPositionRectangularCompositeBuilder::AddAttribute(KContainer* aContainer)
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
    if (aContainer->GetName() == "x") {
        fObject->SetXValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "y") {
        fObject->SetYValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "z") {
        fObject->SetZValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    return false;
}

template<> inline bool KSGenPositionRectangularCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 1) == "x") {
        aContainer->ReleaseTo(fObject, &KSGenPositionRectangularComposite::SetXValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 1) == "y") {
        aContainer->ReleaseTo(fObject, &KSGenPositionRectangularComposite::SetYValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 1) == "z") {
        aContainer->ReleaseTo(fObject, &KSGenPositionRectangularComposite::SetZValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
