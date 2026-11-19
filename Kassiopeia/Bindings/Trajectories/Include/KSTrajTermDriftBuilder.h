#ifndef Kassiopeia_KSTrajTermDriftBuilder_h_
#define Kassiopeia_KSTrajTermDriftBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermDrift.h"

using namespace Kassiopeia;
namespace katrin
{

//! this term in the equation of motion adds e-cross-b and grad-b drifts to adiabatic simulations.
//! @example <kstraj_term_drift name="term_drift"/>
typedef KComplexElement<KSTrajTermDrift> KSTrajTermDriftBuilder;

template<> inline bool KSTrajTermDriftBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
