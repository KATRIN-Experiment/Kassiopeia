#ifndef Kassiopeia_KSTrajIntegratorRKDP54Builder_h_
#define Kassiopeia_KSTrajIntegratorRKDP54Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRKDP54.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajIntegratorRKDP54> KSTrajIntegratorRKDP54Builder;

template<> inline bool KSTrajIntegratorRKDP54Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
