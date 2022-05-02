#ifndef Kassiopeia_KSGenDirectionSphericalMagneticFieldBuilder_h_
#define Kassiopeia_KSGenDirectionSphericalMagneticFieldBuilder_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenDirectionSphericalMagneticField.h"
#include "KToolbox.h"
#include "KSFieldFinder.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

typedef KComplexElement<KSGenDirectionSphericalMagneticField> KSGenDirectionSphericalMagneticFieldBuilder;

template<> inline bool KSGenDirectionSphericalMagneticFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "magnetic_field_name") {
        fObject->AddMagneticField(getMagneticField(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "theta") {
        fObject->SetThetaValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "phi") {
        fObject->SetPhiValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsString()));
        return true;
    }
    return false;
}

template<> inline bool KSGenDirectionSphericalMagneticFieldBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 5) == "theta") {
        aContainer->ReleaseTo(fObject, &KSGenDirectionSphericalMagneticField::SetThetaValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 3) == "phi") {
        aContainer->ReleaseTo(fObject, &KSGenDirectionSphericalMagneticField::SetPhiValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
