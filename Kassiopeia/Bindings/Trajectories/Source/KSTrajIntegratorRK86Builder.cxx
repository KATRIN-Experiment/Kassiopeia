#include "KSTrajIntegratorRK86Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRK86Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRK86Structure =
        KSTrajIntegratorRK86Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRK86 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK86 >( "kstraj_integrator_rk86" );


}
