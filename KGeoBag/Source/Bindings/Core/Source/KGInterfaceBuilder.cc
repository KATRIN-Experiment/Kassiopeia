#include "KGInterfaceBuilder.hh"
#include "KElementProcessor.hh"

namespace katrin
{

    template< >
    KGInterfaceBuilder::~KComplexElement()
    {
    }

    STATICINT sKGInterfaceStructure =
        KGInterfaceBuilder::Attribute< bool >( "reset" );

    STATICINT sKGInterface =
        KElementProcessor::ComplexElement< KGInterface >( "geometry" );

}
