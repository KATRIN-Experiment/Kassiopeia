#include "KSTrajIntegratorRK54Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajIntegratorRK54Builder::~KComplexElement()
    {
    }

    static int sKSTrajIntegratorRK54Structure =
        KSTrajIntegratorRK54Builder::Attribute< string >( "name" );

    static int sToolboxKSTrajIntegratorRK54 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK54 >( "kstraj_integrator_rk54" );

}
