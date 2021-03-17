#ifndef Kassiopeia_KSGenSpinCompositeBuilder_h_
#define Kassiopeia_KSGenSpinCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenSpinComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

typedef KComplexElement<KSGenSpinComposite> KSGenSpinCompositeBuilder;

template<> inline bool KSGenSpinCompositeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surface") {
        KGSurface* tSurface = KGInterface::GetInstance()->RetrieveSurface(aContainer->AsString());
        fObject->SetXAxis(tSurface->GetXAxis());
        fObject->SetYAxis(tSurface->GetYAxis());
        fObject->SetZAxis(tSurface->GetZAxis());
        return true;
    }
    if (aContainer->GetName() == "space") {
        KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace(aContainer->AsString());
        fObject->SetXAxis(tSpace->GetXAxis());
        fObject->SetYAxis(tSpace->GetYAxis());
        fObject->SetZAxis(tSpace->GetZAxis());
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

template<> inline bool KSGenSpinCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 5) == "theta") {
        aContainer->ReleaseTo(fObject, &KSGenSpinComposite::SetThetaValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 3) == "phi") {
        aContainer->ReleaseTo(fObject, &KSGenSpinComposite::SetPhiValue);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
