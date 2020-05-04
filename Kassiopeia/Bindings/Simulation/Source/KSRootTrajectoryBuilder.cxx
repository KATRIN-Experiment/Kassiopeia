#include "KSRootTrajectoryBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootTrajectoryBuilder::~KComplexElement() {}

STATICINT sKSRootTrajectory = KSRootBuilder::ComplexElement<KSRootTrajectory>("ks_root_trajectory");

STATICINT sKSRootTrajectoryStructure =
    KSRootTrajectoryBuilder::Attribute<string>("name") + KSRootTrajectoryBuilder::Attribute<string>("set_trajectory");

}  // namespace katrin
