#ifndef Kassiopeia_KSTrajIntegratorRK87Builder_h_
#define Kassiopeia_KSTrajIntegratorRK87Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK87.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajIntegratorRK87> KSTrajIntegratorRK87Builder;

template<> inline bool KSTrajIntegratorRK87Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
