#ifndef Kassiopeia_KSTrajTermGyrationBuilder_h_
#define Kassiopeia_KSTrajTermGyrationBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermGyration.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajTermGyration> KSTrajTermGyrationBuilder;

template<> inline bool KSTrajTermGyrationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
