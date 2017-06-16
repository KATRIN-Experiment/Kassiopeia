#include "KSTrajIntegratorRK65Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRK65Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRK65Structure =
        KSTrajIntegratorRK65Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRK65 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK65 >( "kstraj_integrator_rk65" );

}
