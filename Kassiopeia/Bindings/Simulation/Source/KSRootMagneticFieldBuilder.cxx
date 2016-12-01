#include "KSRootMagneticFieldBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSRootMagneticFieldBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootMagneticField =
        KSRootBuilder::ComplexElement< KSRootMagneticField >( "ks_root_magnetic_field" );

    STATICINT sKSRootMagneticFieldStructure =
        KSRootMagneticFieldBuilder::Attribute< string >( "name" ) +
        KSRootMagneticFieldBuilder::Attribute< string >( "add_magnetic_field" );

}
