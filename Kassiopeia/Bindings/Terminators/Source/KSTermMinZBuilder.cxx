#include "KSTermMinZBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMinZBuilder::~KComplexElement()
    {
    }

    static int sKSTermMinZStructure =
        KSTermMinZBuilder::Attribute< string >( "name" ) +
        KSTermMinZBuilder::Attribute< double >( "z" );

    static int sKSTermMinZ =
        KSRootBuilder::ComplexElement< KSTermMinZ >( "ksterm_min_z" );


}
