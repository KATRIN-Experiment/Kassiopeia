#ifndef Kassiopeia_KSTrajControlMDotBuilder_h_
#define Kassiopeia_KSTrajControlMDotBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlMDot.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlMDot> KSTrajControlMDotBuilder;

template<> inline bool KSTrajControlMDotBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "fraction") {
        aContainer->CopyTo(fObject, &KSTrajControlMDot::SetFraction);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
