#ifndef Kassiopeia_KSTermMinZBuilder_h_
#define Kassiopeia_KSTermMinZBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMinZ.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose position's z coordinate falls below a given value.
//! @example <ksterm_min_z name="term_min_z" z="1.239"/>
typedef KComplexElement<KSTermMinZ> KSTermMinZBuilder;

template<> inline bool KSTermMinZBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "z") {
        aContainer->CopyTo(fObject, &KSTermMinZ::SetMinZ);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
