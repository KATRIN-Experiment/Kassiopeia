#include "KSFieldElectricConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricConstantBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldElectricConstantStructure =
        KSFieldElectricConstantBuilder::Attribute< string >( "name" ) +
        KSFieldElectricConstantBuilder::Attribute< KThreeVector >( "field" );

    STATICINT sKSFieldElectricConstant =
        KSRootBuilder::ComplexElement< KSFieldElectricConstant >( "ksfield_electric_constant" );

}
