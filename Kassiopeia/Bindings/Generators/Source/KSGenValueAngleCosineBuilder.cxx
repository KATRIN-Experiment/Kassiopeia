#include "KSGenValueAngleCosineBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenValueAngleCosineBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueAngleCosineStructure =
        KSGenValueAngleCosineBuilder::Attribute< string >( "name" ) +
        KSGenValueAngleCosineBuilder::Attribute< double >( "angle_min" ) +
        KSGenValueAngleCosineBuilder::Attribute< double >( "angle_max" );

    STATICINT sToolboxKSGenValueAngleCosine =
        KSRootBuilder::ComplexElement< KSGenValueAngleCosine >( "ksgen_value_angle_cosine" );

}
