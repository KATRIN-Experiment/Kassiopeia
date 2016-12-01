#include "KSGenValueRadiusFractionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenValueRadiusFractionBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueRadiusFractionStructure =
        KSGenValueRadiusFractionBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSGenValueRadiusFraction =
        KSRootBuilder::ComplexElement< KSGenValueRadiusFraction >( "ksgen_value_radius_fraction" );

}
