#include "KSTrajControlCyclotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlCyclotronBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlCyclotronStructure =
        KSTrajControlCyclotronBuilder::Attribute< string >( "name" ) +
        KSTrajControlCyclotronBuilder::Attribute< double >( "fraction" );

    STATICINT sToolboxKSTrajControlCyclotron =
        KSRootBuilder::ComplexElement< KSTrajControlCyclotron >( "kstraj_control_cyclotron" );

}
