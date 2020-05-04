#ifndef Kassiopeia_KSTrajIntegratorRKDP853Builder_h_
#define Kassiopeia_KSTrajIntegratorRKDP853Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRKDP853.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajIntegratorRKDP853> KSTrajIntegratorRKDP853Builder;

template<> inline bool KSTrajIntegratorRKDP853Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
