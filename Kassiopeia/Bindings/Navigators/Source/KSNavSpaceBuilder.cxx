#include "KSNavSpaceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSNavSpaceBuilder::~KComplexElement() = default;

STATICINT sKSNavSpaceStructure =
    //! the name of the navigator.
    KSNavSpaceBuilder::Attribute<std::string>("name") +
    //! if true, the navigator will split tracks when particles enter spaces.
    KSNavSpaceBuilder::Attribute<bool>("enter_split") +
    //! if true, the navigator will split tracks when particles exit spaces.
    KSNavSpaceBuilder::Attribute<bool>("exit_split") +
    //! if true, additional calculations will be performed each step to find failures in the navigation state of the particle
    //! is disabled by default, as it increases the computation time of the simulation slightly
    KSNavSpaceBuilder::Attribute<bool>("fail_check") + KSNavSpaceBuilder::Attribute<double>("tolerance");

STATICINT sToolboxKSNavSpace = KSRootBuilder::ComplexElement<KSNavSpace>("ksnav_space");

}  // namespace katrin
