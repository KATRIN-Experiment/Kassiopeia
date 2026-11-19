#ifndef Kassiopeia_KSTrajTermPropagationBuilder_h_
#define Kassiopeia_KSTrajTermPropagationBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermPropagation.h"

using namespace Kassiopeia;
namespace katrin
{

//! this term in the equation of motion moves the particle in external electromagnetic fields.
//! @example <kstraj_term_propagation name="term_propagation" direction="forward"/>
typedef KComplexElement<KSTrajTermPropagation> KSTrajTermPropagationBuilder;

template<> inline bool KSTrajTermPropagationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "direction") {
        if (aContainer->AsString() == "forward") {
            fObject->SetDirection(KSTrajTermPropagation::eForward);
            return true;
        }
        if (aContainer->AsString() == "backward") {
            fObject->SetDirection(KSTrajTermPropagation::eBackward);
            return true;
        }
        return false;
    }
    return false;
}

}  // namespace katrin
#endif
