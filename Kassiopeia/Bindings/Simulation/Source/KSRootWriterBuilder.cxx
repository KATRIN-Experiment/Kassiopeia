#include "KSRootWriterBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootWriterBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootWriter =
        KSRootBuilder::ComplexElement< KSRootWriter >( "ks_root_writer" );

    STATICINT sKSRootWriterStructure =
        KSRootWriterBuilder::Attribute< string >( "name" ) +
        KSRootWriterBuilder::Attribute< string >( "add_writer" );

}
