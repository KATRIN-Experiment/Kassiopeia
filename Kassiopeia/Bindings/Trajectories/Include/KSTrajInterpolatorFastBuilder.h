#ifndef Kassiopeia_KSTrajInterpolatorFastBuilder_h_
#define Kassiopeia_KSTrajInterpolatorFastBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajInterpolatorFast.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajInterpolatorFast> KSTrajInterpolatorFastBuilder;

template<> inline bool KSTrajInterpolatorFastBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
