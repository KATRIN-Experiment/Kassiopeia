#include "KSFieldElectricQuadrupoleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricQuadrupoleBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldElectricQuadrupoleStructure =
        KSFieldElectricQuadrupoleBuilder::Attribute< string >( "name" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< KThreeVector >( "location" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "strength" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "length" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "radius" );

    STATICINT sKSFieldElectricQuadrupole =
        KSRootBuilder::ComplexElement< KSFieldElectricQuadrupole >( "ksfield_electric_quadrupole" );

}
