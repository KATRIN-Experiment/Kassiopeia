#include "KSWriteASCIIBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSWriteASCIIBuilder::~KComplexElement()
    {
    }

    static int sKSWriteASCIIStructure =
        KSWriteASCIIBuilder::Attribute< string >( "name" ) +
        KSWriteASCIIBuilder::Attribute< string >( "base" ) +
        KSWriteASCIIBuilder::Attribute< string >( "path" )+
        KSWriteASCIIBuilder::Attribute< unsigned int >( "precision" );

    static int sKSWriteASCII =
        KSRootBuilder::ComplexElement< KSWriteASCII >( "kswrite_ascii" );

}
