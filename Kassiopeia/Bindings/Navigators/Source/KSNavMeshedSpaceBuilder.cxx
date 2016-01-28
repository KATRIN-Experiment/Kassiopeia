#include "KSNavMeshedSpaceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSNavMeshedSpaceBuilder::~KComplexElement()
    {
    }

    STATICINT sKSNavMeshedSpaceStructure =
        KSNavMeshedSpaceBuilder::Attribute< string >( "name" ) +
        KSNavMeshedSpaceBuilder::Attribute< string >( "octree_file" ) +
        KSNavMeshedSpaceBuilder::Attribute< bool >( "enter_split" ) +
        KSNavMeshedSpaceBuilder::Attribute< bool >( "exit_split" ) +
        KSNavMeshedSpaceBuilder::Attribute< bool >( "fail_check" ) +
        KSNavMeshedSpaceBuilder::Attribute< string >( "root_space" ) +
        KSNavMeshedSpaceBuilder::Attribute< unsigned int >( "max_octree_depth" ) +
        KSNavMeshedSpaceBuilder::Attribute< double >( "spatial_resolution" ) +
        KSNavMeshedSpaceBuilder::Attribute< unsigned int >( "n_allowed_elements" ) +
        KSNavMeshedSpaceBuilder::Attribute< double >( "absolute_tolerance" ) +
        KSNavMeshedSpaceBuilder::Attribute< double >( "relative_tolerance" );

    STATICINT sToolboxKSNavMeshedSpace =
        KSRootBuilder::ComplexElement< KSNavMeshedSpace >( "ksnav_meshed_space" );

}
