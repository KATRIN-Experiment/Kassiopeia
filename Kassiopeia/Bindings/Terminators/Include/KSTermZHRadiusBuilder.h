#ifndef Kassiopeia_KSTermZHRadiusBuilder_h_
#define Kassiopeia_KSTermZHRadiusBuilder_h_

#include "KComplexElement.hh"
#include "KSTermZHRadius.h"
#include "KSFieldFinder.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermZHRadius> KSTermZHRadiusBuilder;

template<> inline bool KSTermZHRadiusBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "magnetic_field") {
        auto tMagneticField = getMagneticField(aContainer->AsString());
        fObject->AddMagneticField(tMagneticField);
        return true;
    }
    if (aContainer->GetName() == "electric_field") {
        auto tElectricField = getElectricField(aContainer->AsString());
        fObject->AddElectricField(tElectricField);
        return true;
    }
    if (aContainer->GetName() == "central_expansion") {
        aContainer->CopyTo(fObject, &KSTermZHRadius::SetCheckCentralExpansion);
        return true;
    }
    if (aContainer->GetName() == "remote_expansion") {
        aContainer->CopyTo(fObject, &KSTermZHRadius::SetCheckRemoteExpansion);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
