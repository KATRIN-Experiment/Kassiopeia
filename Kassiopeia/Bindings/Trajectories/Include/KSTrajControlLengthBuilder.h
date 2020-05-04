#ifndef Kassiopeia_KSTrajControlLengthBuilder_h_
#define Kassiopeia_KSTrajControlLengthBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlLength.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlLength> KSTrajControlLengthBuilder;

template<> inline bool KSTrajControlLengthBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "length") {
        aContainer->CopyTo(fObject, &KSTrajControlLength::SetLength);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
