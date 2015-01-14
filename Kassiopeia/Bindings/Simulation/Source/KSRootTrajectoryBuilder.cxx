#include "KSRootTrajectoryBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootTrajectoryBuilder::~KComplexElement()
    {
    }

    static const int sKSRootTrajectory =
        KSRootBuilder::ComplexElement< KSRootTrajectory >( "ks_root_trajectory" );

    static const int sKSRootTrajectoryStructure =
        KSRootTrajectoryBuilder::Attribute< string >( "name" ) +
        KSRootTrajectoryBuilder::Attribute< string >( "set_trajectory" );

}
