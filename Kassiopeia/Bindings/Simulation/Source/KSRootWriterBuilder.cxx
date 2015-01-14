#include "KSRootWriterBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootWriterBuilder::~KComplexElement()
    {
    }

    static int sKSRootWriter =
        KSRootBuilder::ComplexElement< KSRootWriter >( "ks_root_writer" );

    static int sKSRootWriterStructure =
        KSRootWriterBuilder::Attribute< string >( "name" ) +
        KSRootWriterBuilder::Attribute< string >( "add_writer" );

}
