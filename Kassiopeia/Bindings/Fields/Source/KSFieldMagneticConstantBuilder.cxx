#include "KSFieldMagneticConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticConstantBuilder::~KComplexElement()
    {
    }

    static int sKSFieldMagneticConstantStructure =
        KSFieldMagneticConstantBuilder::Attribute< string >( "name" ) +
        KSFieldMagneticConstantBuilder::Attribute< KThreeVector >( "field" );

    static int sKSFieldMagneticConstant =
        KSRootBuilder::ComplexElement< KSFieldMagneticConstant >( "ksfield_magnetic_constant" );

}
