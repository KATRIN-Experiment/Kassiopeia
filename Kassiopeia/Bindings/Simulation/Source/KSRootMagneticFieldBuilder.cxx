#include "KSRootMagneticFieldBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootMagneticFieldBuilder::~KComplexElement()
    {
    }

    static int sKSRootMagneticField =
        KSRootBuilder::ComplexElement< KSRootMagneticField >( "ks_root_magnetic_field" );

    static int sKSRootMagneticFieldStructure =
        KSRootMagneticFieldBuilder::Attribute< string >( "name" ) +
        KSRootMagneticFieldBuilder::Attribute< string >( "add_magnetic_field" );

}
