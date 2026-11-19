#ifndef Kassiopeia_KSTrajIntegratorRK8Builder_h_
#define Kassiopeia_KSTrajIntegratorRK8Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK8.h"

using namespace Kassiopeia;
namespace katrin
{

//! an 8th order runge-kutta integrator.
//! this is the second most accurate and slowes integrator in kassiopeia.
//! this is also most common choice.
//! @example <kstraj_integrator_rk8 name="integrator_rk8"/>
typedef KComplexElement<KSTrajIntegratorRK8> KSTrajIntegratorRK8Builder;

template<> inline bool KSTrajIntegratorRK8Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
