#include "KGInterfaceBuilder.hh"
#include "KElementProcessor.hh"

namespace katrin
{

    template< >
    KGInterfaceBuilder::~KComplexElement()
    {
    }

    static const int sKGInterfaceStructure =
        KGInterfaceBuilder::Attribute< bool >( "reset" );

    static const int sKGInterface =
        KElementProcessor::ComplexElement< KGInterface >( "geometry" );

}
