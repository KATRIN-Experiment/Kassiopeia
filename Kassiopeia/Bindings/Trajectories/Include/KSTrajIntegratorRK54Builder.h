#ifndef Kassiopeia_KSTrajIntegratorRK54Builder_h_
#define Kassiopeia_KSTrajIntegratorRK54Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK54.h"

using namespace Kassiopeia;
namespace katrin
{

//! an 5th/4th order embedded runge-kutta integrator.
//! this is the fastest and least accurate integrator in kassiopeia.
//! @example <kstraj_integrator_rk54 name="integrator_rk54"/>
typedef KComplexElement<KSTrajIntegratorRK54> KSTrajIntegratorRK54Builder;

template<> inline bool KSTrajIntegratorRK54Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
