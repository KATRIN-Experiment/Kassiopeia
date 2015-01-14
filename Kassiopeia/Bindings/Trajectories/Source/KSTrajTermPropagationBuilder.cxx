#include "KSTrajTermPropagationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermPropagationBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTermPropagationStructure =
        KSTrajTermPropagationBuilder::Attribute< string >( "name" ) +
        KSTrajTermPropagationBuilder::Attribute< string >( "direction" );

    static int sToolboxKSTrajTermPropagation =
        KSRootBuilder::ComplexElement< KSTrajTermPropagation >( "kstraj_term_propagation" );

}
