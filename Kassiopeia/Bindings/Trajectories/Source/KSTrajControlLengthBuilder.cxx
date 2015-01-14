#include "KSTrajControlLengthBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlLengthBuilder::~KComplexElement()
    {
    }

    static int sKSTrajControlLengthStructure =
        KSTrajControlLengthBuilder::Attribute< string >( "name" ) +
        KSTrajControlLengthBuilder::Attribute< double >( "length" );

    static int sToolboxKSTrajControlLength =
        KSRootBuilder::ComplexElement< KSTrajControlLength >( "kstraj_control_length" );

}
