#include "KSTrajControlBChangeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajControlBChangeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlBChangeStructure =
        KSTrajControlBChangeBuilder::Attribute< string >( "name" ) +
        KSTrajControlBChangeBuilder::Attribute< double >( "fraction" );

    STATICINT sToolboxKSTrajControlBChange =
        KSRootBuilder::ComplexElement< KSTrajControlBChange >( "kstraj_control_B_change" );

}
