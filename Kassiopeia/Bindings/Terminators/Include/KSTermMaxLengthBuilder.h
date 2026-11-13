#ifndef Kassiopeia_KSTermMaxLengthBuilder_h_
#define Kassiopeia_KSTermMaxLengthBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxLength.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills particles whose track length rises above a given value.
//! @example <ksterm_max_length name="term_max_length" length="1.239"/>
typedef KComplexElement<KSTermMaxLength> KSTermMaxLengthBuilder;

template<> inline bool KSTermMaxLengthBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "length") {
        aContainer->CopyTo(fObject, &KSTermMaxLength::SetLength);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
