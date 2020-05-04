#ifndef Kassiopeia_KSTrajTermGravityBuilder_h_
#define Kassiopeia_KSTrajTermGravityBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermGravity.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajTermGravity> KSTrajTermGravityBuilder;

template<> inline bool KSTrajTermGravityBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "gravity") {
        aContainer->CopyTo(fObject, &KSTrajTermGravity::SetGravity);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
