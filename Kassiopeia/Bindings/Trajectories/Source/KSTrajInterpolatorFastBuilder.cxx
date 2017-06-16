#include "KSTrajInterpolatorFastBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajInterpolatorFastBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajInterpolatorFastStructure =
        KSTrajInterpolatorFastBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajInterpolatorFast =
        KSRootBuilder::ComplexElement< KSTrajInterpolatorFast >( "kstraj_interpolator_fast" );

}
