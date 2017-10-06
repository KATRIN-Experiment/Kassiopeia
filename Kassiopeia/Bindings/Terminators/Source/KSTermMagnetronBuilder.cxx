#include "KSTermMagnetronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMagnetronBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMagnetronStructure =
        KSTermMagnetronBuilder::Attribute< string >( "name" ) +
        KSTermMagnetronBuilder::Attribute< double >( "max_phi" );

    STATICINT sKSTermMagnetron =
        KSRootBuilder::ComplexElement< KSTermMagnetron >( "ksterm_magnetron" );

}
