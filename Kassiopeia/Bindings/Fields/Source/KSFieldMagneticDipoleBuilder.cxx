#include "KSFieldMagneticDipoleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticDipoleBuilder::~KComplexElement()
    {
    }

    static int sKSFieldMagneticDipoleStructure =
        KSFieldMagneticDipoleBuilder::Attribute< string >( "name" ) +
        KSFieldMagneticDipoleBuilder::Attribute< KThreeVector >( "location" ) +
        KSFieldMagneticDipoleBuilder::Attribute< KThreeVector >( "moment" );

    static int sKSFieldMagneticDipole =
        KSRootBuilder::ComplexElement< KSFieldMagneticDipole >( "ksfield_magnetic_dipole" );

}
