#include "KSFieldElectricPotentialmapBuilder.h"
#include "KSFieldElectricConstantBuilder.h"
#include "KSFieldElectricQuadrupoleBuilder.h"
#include "KSFieldElectrostaticBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricPotentialmapBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldElectricPotentialmapStructure =
        KSFieldElectricPotentialmapBuilder::Attribute< string >( "name" ) +
        KSFieldElectricPotentialmapBuilder::Attribute< string >( "directory" ) +
        KSFieldElectricPotentialmapBuilder::Attribute< string >( "file" ) +
        KSFieldElectricPotentialmapBuilder::Attribute< string >( "interpolation" );

    STATICINT sKSFieldElectricPotentialmap =
        KSRootBuilder::ComplexElement< KSFieldElectricPotentialmap >( "ksfield_electric_potentialmap" );

    ////////////////////////////////////////////////////////////////////

    template< >
    KSFieldElectricPotentialmapCalculatorBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldElectricPotentialmapCalculatorStructure =
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< string >( "directory" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< string >( "file" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< KThreeVector >( "center" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< KThreeVector >( "length" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_x" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_y" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< bool >( "mirror_z" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< double >( "spacing" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::ComplexElement< KSFieldElectricConstant >( "field_electric_constant" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::ComplexElement< KSFieldElectricQuadrupole >( "field_electric_quadrupole" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::ComplexElement< KSFieldElectrostatic >( "field_electrostatic" ) +
        KSFieldElectricPotentialmapCalculatorBuilder::Attribute< string >( "spaces" );

    STATICINT sKSFieldElectricPotentialmapCalculator =
        KSRootBuilder::ComplexElement< KSFieldElectricPotentialmapCalculator >( "ksfield_electric_potentialmap_calculator" );

}
