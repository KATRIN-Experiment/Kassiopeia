#include "KSRootElectricFieldBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootElectricFieldBuilder::~KComplexElement()
    {
    }

    static const int sKSRootElectricField =
        KSRootBuilder::ComplexElement< KSRootElectricField >( "ks_root_electric_field" );

    static const int sKSRootElectricFieldStructure =
        KSRootElectricFieldBuilder::Attribute< string >( "name" ) +
        KSRootElectricFieldBuilder::Attribute< string >( "add_electric_field" );

}
