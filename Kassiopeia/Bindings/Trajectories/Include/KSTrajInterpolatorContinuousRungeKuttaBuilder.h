#ifndef Kassiopeia_KSTrajInterpolatorContinuousRungeKuttaBuilder_h_
#define Kassiopeia_KSTrajInterpolatorContinuousRungeKuttaBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajInterpolatorContinuousRungeKutta.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajInterpolatorContinuousRungeKutta > KSTrajInterpolatorContinuousRungeKuttaBuilder;

    template< >
    inline bool KSTrajInterpolatorContinuousRungeKuttaBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }

}

#endif
