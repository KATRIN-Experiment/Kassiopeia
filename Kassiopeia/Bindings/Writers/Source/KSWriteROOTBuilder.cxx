#include "KSWriteROOTBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSWriteROOTBuilder::~KComplexElement()
    {
    }

    STATICINT sKSWriteROOTStructure =
        KSWriteROOTBuilder::Attribute< string >( "name" ) +
        KSWriteROOTBuilder::Attribute< string >( "base" ) +
        KSWriteROOTBuilder::Attribute< string >( "path" );

    STATICINT sKSWriteROOT =
        KSRootBuilder::ComplexElement< KSWriteROOT >( "kswrite_root" );

}
