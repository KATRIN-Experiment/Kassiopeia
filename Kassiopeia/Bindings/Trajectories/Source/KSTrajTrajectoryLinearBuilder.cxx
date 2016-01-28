#include "KSTrajTrajectoryLinearBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTrajectoryLinearBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTrajectoryLinearStructure =
        KSTrajTrajectoryLinearBuilder::Attribute< string >( "name" )+
        KSTrajTrajectoryLinearBuilder::Attribute< double >( "length" );

    STATICINT sKSTrajTrajectoryLinear =
        KSRootBuilder::ComplexElement < KSTrajTrajectoryLinear >( "kstraj_trajectory_linear" );

}
