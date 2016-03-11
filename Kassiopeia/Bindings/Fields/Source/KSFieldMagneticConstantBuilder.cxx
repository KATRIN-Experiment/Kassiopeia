#include "KSFieldMagneticConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticConstantBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldMagneticConstantStructure =
        KSFieldMagneticConstantBuilder::Attribute< string >( "name" ) +
        KSFieldMagneticConstantBuilder::Attribute< KThreeVector >( "field" );

    STATICINT sKSFieldMagneticConstant =
        KSRootBuilder::ComplexElement< KSFieldMagneticConstant >( "ksfield_magnetic_constant" );

}
