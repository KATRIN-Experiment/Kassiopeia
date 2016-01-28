#include "KSTrajInterpolatorContinuousRungeKuttaBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajInterpolatorContinuousRungeKuttaBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajInterpolatorContinuousRungeKuttaStructure =
        KSTrajInterpolatorContinuousRungeKuttaBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajInterpolatorContinuousRungeKutta =
        KSRootBuilder::ComplexElement< KSTrajInterpolatorContinuousRungeKutta >( "kstraj_interpolator_crk" );

}
