#include "KSTrajInterpolatorHermiteBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

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
