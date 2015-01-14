#include "KSTrajIntegratorRK87Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajIntegratorRK87Builder::~KComplexElement()
    {
    }

    static int sKSTrajIntegratorRK87Structure =
        KSTrajIntegratorRK87Builder::Attribute< string >( "name" );

    static int sToolboxKSTrajIntegratorRK87 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK87 >( "kstraj_integrator_rk87" );


}
