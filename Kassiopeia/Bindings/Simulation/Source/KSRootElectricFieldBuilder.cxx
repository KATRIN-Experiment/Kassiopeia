#include "KSRootElectricFieldBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootElectricFieldBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootElectricField =
        KSRootBuilder::ComplexElement< KSRootElectricField >( "ks_root_electric_field" );

    STATICINT sKSRootElectricFieldStructure =
        KSRootElectricFieldBuilder::Attribute< string >( "name" ) +
        KSRootElectricFieldBuilder::Attribute< string >( "add_electric_field" );

}
