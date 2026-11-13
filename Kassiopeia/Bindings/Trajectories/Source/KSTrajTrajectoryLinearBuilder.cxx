#include "KSTrajTrajectoryLinearBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTrajectoryLinearBuilder::~KComplexElement() = default;

STATICINT sKSTrajTrajectoryLinearStructure =
    //! the name of this trajectory.
    KSTrajTrajectoryLinearBuilder::Attribute<std::string>("name") +
    //! the length of the resulting trajectory segment in meters.
    KSTrajTrajectoryLinearBuilder::Attribute<double>("length");

STATICINT sKSTrajTrajectoryLinear = KSRootBuilder::ComplexElement<KSTrajTrajectoryLinear>("kstraj_trajectory_linear");

}  // namespace katrin
