#include "KSTrajControlTimeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlTimeBuilder::~KComplexElement()
    {
    }

    static int sKSTrajControlTimeStructure =
        KSTrajControlTimeBuilder::Attribute< string >( "name" ) +
        KSTrajControlTimeBuilder::Attribute< double >( "time" );

    static int sToolboxKSTrajControlTime =
        KSRootBuilder::ComplexElement< KSTrajControlTime >( "kstraj_control_time" );

}
