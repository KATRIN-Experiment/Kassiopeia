#include "KSWriteROOTConditionStepBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionStepBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionStepStructure =
    //! the name of this write condition.
    KSWriteROOTConditionStepBuilder::Attribute<std::string>("name") +
    //! the fraction of steps to be recorded
    KSWriteROOTConditionStepBuilder::Attribute<int>("nth_step") +
    //! the name of the group the output variable is in (optional)
    KSWriteROOTConditionStepBuilder::Attribute<std::string>("group") +
    //! the name of the output variable which is used for the write decision (must be of type int)
    KSWriteROOTConditionStepBuilder::Attribute<std::string>("parent");


STATICINT sKSWriteROOTConditionStep =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionStepData>("kswrite_root_condition_step");

}  // namespace katrin
