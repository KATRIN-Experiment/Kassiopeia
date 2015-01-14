#include "KSTrajIntegratorRK65Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajIntegratorRK65Builder::~KComplexElement()
    {
    }

    static int sKSTrajIntegratorRK65Structure =
        KSTrajIntegratorRK65Builder::Attribute< string >( "name" );

    static int sToolboxKSTrajIntegratorRK65 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK65 >( "kstraj_integrator_rk65" );

}
