#include "KSTermMinZBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermMinZBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMinZStructure =
        KSTermMinZBuilder::Attribute< string >( "name" ) +
        KSTermMinZBuilder::Attribute< double >( "z" );

    STATICINT sKSTermMinZ =
        KSRootBuilder::ComplexElement< KSTermMinZ >( "ksterm_min_z" );


}
