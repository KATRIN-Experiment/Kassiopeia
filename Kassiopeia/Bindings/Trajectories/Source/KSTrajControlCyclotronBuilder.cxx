#include "KSTrajControlCyclotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlCyclotronBuilder::~KComplexElement()
    {
    }

    static int sKSTrajControlCyclotronStructure =
        KSTrajControlCyclotronBuilder::Attribute< string >( "name" ) +
        KSTrajControlCyclotronBuilder::Attribute< double >( "fraction" );

    static int sToolboxKSTrajControlCyclotron =
        KSRootBuilder::ComplexElement< KSTrajControlCyclotron >( "kstraj_control_cyclotron" );

}
