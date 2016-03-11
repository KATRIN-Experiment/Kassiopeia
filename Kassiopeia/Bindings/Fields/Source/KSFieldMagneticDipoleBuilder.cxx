#include "KSFieldMagneticDipoleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticDipoleBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldMagneticDipoleStructure =
        KSFieldMagneticDipoleBuilder::Attribute< string >( "name" ) +
        KSFieldMagneticDipoleBuilder::Attribute< KThreeVector >( "location" ) +
        KSFieldMagneticDipoleBuilder::Attribute< KThreeVector >( "moment" );

    STATICINT sKSFieldMagneticDipole =
        KSRootBuilder::ComplexElement< KSFieldMagneticDipole >( "ksfield_magnetic_dipole" );

}
