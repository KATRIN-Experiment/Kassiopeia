#include "KSTrajInterpolatorFastBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajInterpolatorFastBuilder::~KComplexElement()
    {
    }

    static int sKSTrajInterpolatorFastStructure =
        KSTrajInterpolatorFastBuilder::Attribute< string >( "name" );

    static int sToolboxKSTrajInterpolatorFast =
        KSRootBuilder::ComplexElement< KSTrajInterpolatorFast >( "kstraj_interpolator_fast" );

}
