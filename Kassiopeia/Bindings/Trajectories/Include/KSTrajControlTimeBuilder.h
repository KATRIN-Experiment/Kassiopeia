#ifndef Kassiopeia_KSTrajControlTimeBuilder_h_
#define Kassiopeia_KSTrajControlTimeBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlTime.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlTime> KSTrajControlTimeBuilder;

template<> inline bool KSTrajControlTimeBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "time") {
        aContainer->CopyTo(fObject, &KSTrajControlTime::SetTime);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
