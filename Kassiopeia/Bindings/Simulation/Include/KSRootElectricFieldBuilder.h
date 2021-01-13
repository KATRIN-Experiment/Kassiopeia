#ifndef Kassiopeia_KSRootElectricFieldBuilder_h_
#define Kassiopeia_KSRootElectricFieldBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldFinder.h"
#include "KSRootElectricField.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSRootElectricField> KSRootElectricFieldBuilder;

template<> inline bool KSRootElectricFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "add_electric_field") {
        fObject->AddElectricField(getElectricField(aContainer->AsString()));
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
