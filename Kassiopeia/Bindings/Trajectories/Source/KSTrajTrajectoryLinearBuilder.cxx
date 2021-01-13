#include "KSTrajTrajectoryLinearBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTrajectoryLinearBuilder::~KComplexElement() = default;

STATICINT sKSTrajTrajectoryLinearStructure = KSTrajTrajectoryLinearBuilder::Attribute<std::string>("name") +
                                             KSTrajTrajectoryLinearBuilder::Attribute<double>("length");

STATICINT sKSTrajTrajectoryLinear = KSRootBuilder::ComplexElement<KSTrajTrajectoryLinear>("kstraj_trajectory_linear");

}  // namespace katrin
