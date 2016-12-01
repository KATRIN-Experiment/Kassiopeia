#include "KSTrajControlLengthBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajControlLengthBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlLengthStructure =
        KSTrajControlLengthBuilder::Attribute< string >( "name" ) +
        KSTrajControlLengthBuilder::Attribute< double >( "length" );

    STATICINT sToolboxKSTrajControlLength =
        KSRootBuilder::ComplexElement< KSTrajControlLength >( "kstraj_control_length" );

}
