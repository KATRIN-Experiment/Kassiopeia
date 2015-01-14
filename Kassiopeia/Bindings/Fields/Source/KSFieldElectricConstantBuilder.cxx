#include "KSFieldElectricConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricConstantBuilder::~KComplexElement()
    {
    }

    static int sKSFieldElectricConstantStructure =
        KSFieldElectricConstantBuilder::Attribute< string >( "name" ) +
        KSFieldElectricConstantBuilder::Attribute< KThreeVector >( "field" );

    static int sKSFieldElectricConstant =
        KSRootBuilder::ComplexElement< KSFieldElectricConstant >( "ksfield_electric_constant" );

}
