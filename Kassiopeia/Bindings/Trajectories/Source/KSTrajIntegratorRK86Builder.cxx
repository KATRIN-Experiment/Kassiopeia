#include "KSTrajIntegratorRK86Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajIntegratorRK86Builder::~KComplexElement()
    {
    }

    static int sKSTrajIntegratorRK86Structure =
        KSTrajIntegratorRK86Builder::Attribute< string >( "name" );

    static int sToolboxKSTrajIntegratorRK86 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorRK86 >( "kstraj_integrator_rk86" );


}
