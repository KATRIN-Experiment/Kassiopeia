#include "KSTrajIntegratorRKDP54Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorRKDP54Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorRKDP54Structure =
        KSTrajIntegratorRKDP54Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorRKDP54 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRKDP54 >( "kstraj_integrator_rkdp54" );

}
