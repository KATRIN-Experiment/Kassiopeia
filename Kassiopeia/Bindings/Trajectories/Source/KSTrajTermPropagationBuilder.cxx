#include "KSTrajTermPropagationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajTermPropagationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTermPropagationStructure =
        KSTrajTermPropagationBuilder::Attribute< string >( "name" ) +
        KSTrajTermPropagationBuilder::Attribute< string >( "direction" );

    STATICINT sToolboxKSTrajTermPropagation =
        KSRootBuilder::ComplexElement< KSTrajTermPropagation >( "kstraj_term_propagation" );

}
