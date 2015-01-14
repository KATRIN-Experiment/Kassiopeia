#include "KSTrajIntegratorRK8Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajIntegratorRK8Builder::~KComplexElement()
    {
    }

    static int sKSTrajIntegratorRK8Structure =
        KSTrajIntegratorRK8Builder::Attribute< string >( "name" );

    static int sToolboxKSTrajIntegratorRK8 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK8 >( "kstraj_integrator_rk8" );

}
