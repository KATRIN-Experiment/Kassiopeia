#include "KSTrajIntegratorRKDP853Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRKDP853Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRKDP853Structure =
        KSTrajIntegratorRKDP853Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRKDP853 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRKDP853 >( "kstraj_integrator_rkdp853" );

}
