#ifndef Kassiopeia_KSRootGeneratorBuilder_h_
#define Kassiopeia_KSRootGeneratorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootGenerator.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSRootGenerator> KSRootGeneratorBuilder;

template<> inline bool KSRootGeneratorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "set_generator") {
        fObject->SetGenerator(KToolbox::GetInstance().Get<KSGenerator>(aContainer->AsString()));
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
