#include "KSGenValueRadiusCylindricalBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueRadiusCylindricalBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueRadiusCylindricalStructure =
        KSGenValueRadiusCylindricalBuilder::Attribute< string >( "name" ) +
        KSGenValueRadiusCylindricalBuilder::Attribute< double >( "radius_min" ) +
        KSGenValueRadiusCylindricalBuilder::Attribute< double >( "radius_max" );

    static int sToolboxKSGenValueRadiusCylindrical =
        KSRootBuilder::ComplexElement< KSGenValueRadiusCylindrical >( "ksgen_value_radius_cylindrical" );

}
