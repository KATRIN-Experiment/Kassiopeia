#ifndef KSGENMOMENTUMRECTANGULARCOMPOSITEBUILDER_H
#define KSGENMOMENTUMRECTANGULARCOMPOSITEBUILDER_H

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenMomentumRectangularComposite.h"
#include "KToolbox.h"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

typedef KComplexElement<KSGenMomentumRectangularComposite> KSGenMomentumRectangularCompositeBuilder;

template<> inline bool KSGenMomentumRectangularCompositeBuilder::AddAttribute(KContainer* aContainer)
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
    if (aContainer->GetName() == "x") {
        fObject->SetXValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsReference<std::string>()));
        return true;
    }
    if (aContainer->GetName() == "y") {
        fObject->SetYValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsReference<std::string>()));
        return true;
    }
    if (aContainer->GetName() == "z") {
        fObject->SetZValue(KToolbox::GetInstance().Get<KSGenValue>(aContainer->AsReference<std::string>()));
        return true;
    }
    return false;
}

template<> inline bool KSGenMomentumRectangularCompositeBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->GetName().substr(0, 1) == "x") {
        aContainer->ReleaseTo(fObject, &KSGenMomentumRectangularComposite::SetXValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 1) == "y") {
        aContainer->ReleaseTo(fObject, &KSGenMomentumRectangularComposite::SetYValue);
        return true;
    }
    if (aContainer->GetName().substr(0, 1) == "z") {
        aContainer->ReleaseTo(fObject, &KSGenMomentumRectangularComposite::SetZValue);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif  // KSGENMOMENTUMRECTANGULARCOMPOSITEBUILDER_H
