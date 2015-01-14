#include "KSWriteROOTBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSWriteROOTBuilder::~KComplexElement()
    {
    }

    static int sKSWriteROOTStructure =
        KSWriteROOTBuilder::Attribute< string >( "name" ) +
        KSWriteROOTBuilder::Attribute< string >( "base" ) +
        KSWriteROOTBuilder::Attribute< string >( "path" );

    static int sKSWriteROOT =
        KSRootBuilder::ComplexElement< KSWriteROOT >( "kswrite_root" );

}
