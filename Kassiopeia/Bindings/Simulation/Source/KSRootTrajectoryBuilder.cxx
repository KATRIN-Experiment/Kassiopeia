#include "KSRootTrajectoryBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootTrajectoryBuilder::~KComplexElement() = default;

STATICINT sKSRootTrajectory = KSRootBuilder::ComplexElement<KSRootTrajectory>("ks_root_trajectory");

STATICINT sKSRootTrajectoryStructure = KSRootTrajectoryBuilder::Attribute<std::string>("name") +
                                       KSRootTrajectoryBuilder::Attribute<std::string>("set_trajectory");

}  // namespace katrin
