#include "KSTrajControlTimeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlTimeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlTimeStructure =
        KSTrajControlTimeBuilder::Attribute< string >( "name" ) +
        KSTrajControlTimeBuilder::Attribute< double >( "time" );

    STATICINT sToolboxKSTrajControlTime =
        KSRootBuilder::ComplexElement< KSTrajControlTime >( "kstraj_control_time" );

}
