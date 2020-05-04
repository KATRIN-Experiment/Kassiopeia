#ifndef Kassiopeia_KSGenValueRadiusFractionBuilder_h_
#define Kassiopeia_KSGenValueRadiusFractionBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueRadiusFraction.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueRadiusFraction> KSGenValueRadiusFractionBuilder;

template<> inline bool KSGenValueRadiusFractionBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
