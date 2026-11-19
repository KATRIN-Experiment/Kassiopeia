#ifndef Kassiopeia_KSTrajIntegratorRK65Builder_h_
#define Kassiopeia_KSTrajIntegratorRK65Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK65.h"

using namespace Kassiopeia;
namespace katrin
{

//! an 6th/5th order embedded runge-kutta integrator.
//! @example <kstraj_integrator_rk65 name="integrator_rk65"/>
typedef KComplexElement<KSTrajIntegratorRK65> KSTrajIntegratorRK65Builder;

template<> inline bool KSTrajIntegratorRK65Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
