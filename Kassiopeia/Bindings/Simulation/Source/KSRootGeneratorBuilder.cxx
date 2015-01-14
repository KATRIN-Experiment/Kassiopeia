#include "KSRootGeneratorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootGeneratorBuilder::~KComplexElement()
    {
    }

    static const int sKSRootGenerator =
        KSRootBuilder::ComplexElement< KSRootGenerator >( "ks_root_generator" );

    static const int sKSRootGeneratorStructure =
        KSRootGeneratorBuilder::Attribute< string >( "name" ) +
        KSRootGeneratorBuilder::Attribute< string >( "set_generator" );

}
