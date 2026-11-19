#ifndef Kassiopeia_KSTrajControlLengthBuilder_h_
#define Kassiopeia_KSTrajControlLengthBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlLength.h"

using namespace Kassiopeia;
namespace katrin
{

//! this control suggests a temporal step which would result in a step with the given length were the particle to remain at constant speed.
//! it never rejects steps.
//! @example <kstraj_control_length name="control_length" length="1.e-10"/>
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
