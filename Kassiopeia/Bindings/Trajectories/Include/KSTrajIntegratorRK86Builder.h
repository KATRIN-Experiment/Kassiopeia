#ifndef Kassiopeia_KSTrajIntegratorRK86Builder_h_
#define Kassiopeia_KSTrajIntegratorRK86Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK86.h"

using namespace Kassiopeia;
namespace katrin
{

//! an 8th/6th order embedded runge-kutta integrator.
//! @example <kstraj_integrator_rk86 name="integrator_rk86"/>
typedef KComplexElement<KSTrajIntegratorRK86> KSTrajIntegratorRK86Builder;

template<> inline bool KSTrajIntegratorRK86Builder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
