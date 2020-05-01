#ifndef Kassiopeia_KSGenDirectionSphericalCompositeBuilder_h_
#define Kassiopeia_KSGenDirectionSphericalCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenDirectionSphericalComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

typedef KComplexElement<KSGenDirectionSphericalComposite> KSGenDirectionSphericalCompositeBuilder;

template<> inline bool KSGenDirectionSphericalCompositeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surface") {
        KGSurface* tSurface = KGInterface::GetInstance()->RetrieveSurface(aContainer->AsReference<std::string>());
        fObject->SetXAxis(tSurface->GetXAxis());
        fObject->SetYAxis(tSurface->GetYAxis());
        fObject->SetZAxis(tSurface->GetZAxis());
        return true;
    }
    if (aContainer->GetName() == "space") {
        KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace(aContainer->AsReference<std::string>());
        fObject->SetXAxis(tSpace->GetXAxis());
        fObject->SetYAxis(tSpace->GetYAxis());
        fObject->SetZAxis(tSpace->GetZAxis());
        return true;
    }
    if (aContainer->GetName() == "theta") {
        fObject->SetThetaValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsReference<std::string>()));
        return true;
    }
    if (aContainer->GetName() == "phi") {
        fObject->SetPhiValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsReference<std::string>()));
        return true;
    }
    return false;
}

template<> inline bool KSGenDirectionSphericalCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 5) == "theta") {
        aContainer->ReleaseTo(fObject, &KSGenDirectionSphericalComposite::SetThetaValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 3) == "phi") {
        aContainer->ReleaseTo(fObject, &KSGenDirectionSphericalComposite::SetPhiValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
