#include "KSTrajIntegratorRK54Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRK54Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRK54Structure =
        KSTrajIntegratorRK54Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRK54 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK54 >( "kstraj_integrator_rk54" );

}
