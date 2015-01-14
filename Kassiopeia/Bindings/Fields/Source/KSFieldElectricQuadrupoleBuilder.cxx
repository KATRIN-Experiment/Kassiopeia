#include "KSFieldElectricQuadrupoleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricQuadrupoleBuilder::~KComplexElement()
    {
    }

    static int sKSFieldElectricQuadrupoleStructure =
        KSFieldElectricQuadrupoleBuilder::Attribute< string >( "name" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< KThreeVector >( "location" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "strength" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "length" ) +
        KSFieldElectricQuadrupoleBuilder::Attribute< double >( "radius" );

    static int sKSFieldElectricQuadrupole =
        KSRootBuilder::ComplexElement< KSFieldElectricQuadrupole >( "ksfield_electric_quadrupole" );

}
