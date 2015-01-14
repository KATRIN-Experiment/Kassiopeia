#include "KSTrajTrajectoryLinearBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTrajectoryLinearBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTrajectoryLinearStructure =
        KSTrajTrajectoryLinearBuilder::Attribute< string >( "name" )+
        KSTrajTrajectoryLinearBuilder::Attribute< double >( "length" );

    static int sKSTrajTrajectoryLinear =
        KSRootBuilder::ComplexElement < KSTrajTrajectoryLinear >( "kstraj_trajectory_linear" );

}
