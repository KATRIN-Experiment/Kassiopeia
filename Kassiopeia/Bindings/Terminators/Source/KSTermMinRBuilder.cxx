#include "KSTermMinRBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMinRBuilder::~KComplexElement()
    {
    }

    static int sKSTermMinRStructure =
        KSTermMinRBuilder::Attribute< string >( "name" ) +
        KSTermMinRBuilder::Attribute< double >( "r" );

    static int sKSTermMinR =
        KSRootBuilder::ComplexElement< KSTermMinR >( "ksterm_min_r" );

}
