#ifndef Kassiopeia_KSTrajInterpolatorHermiteBuilder_h_
#define Kassiopeia_KSTrajInterpolatorHermiteBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajInterpolatorHermite.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajInterpolatorHermite> KSTrajInterpolatorHermiteBuilder;

template<> inline bool KSTrajInterpolatorHermiteBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
