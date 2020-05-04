#ifndef Kassiopeia_KSTrajControlCyclotronBuilder_h_
#define Kassiopeia_KSTrajControlCyclotronBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlCyclotron.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlCyclotron> KSTrajControlCyclotronBuilder;

template<> inline bool KSTrajControlCyclotronBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "fraction") {
        aContainer->CopyTo(fObject, &KSTrajControlCyclotron::SetFraction);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
