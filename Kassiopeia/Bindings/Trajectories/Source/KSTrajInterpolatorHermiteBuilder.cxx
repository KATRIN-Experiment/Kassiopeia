#include "KSTrajInterpolatorHermiteBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajInterpolatorHermiteBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajInterpolatorHermiteStructure =
        KSTrajInterpolatorHermiteBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajInterpolatorHermite =
        KSRootBuilder::ComplexElement< KSTrajInterpolatorHermite >( "kstraj_interpolator_hermite" );

}
