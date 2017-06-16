#include "KSTrajIntegratorRK8Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRK8Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRK8Structure =
        KSTrajIntegratorRK8Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRK8 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK8 >( "kstraj_integrator_rk8" );

}
