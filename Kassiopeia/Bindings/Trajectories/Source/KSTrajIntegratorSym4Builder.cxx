#include "KSTrajIntegratorSym4Builder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajIntegratorSym4Builder::~KComplexElement()
    {
    }

    STATICINT sKSTrajIntegratorSym4Structure =
        KSTrajIntegratorSym4Builder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajIntegratorSym4 =
        KSRootBuilder::ComplexElement< KSTrajIntegratorSym4 >( "kstraj_integrator_sym4" );

}
