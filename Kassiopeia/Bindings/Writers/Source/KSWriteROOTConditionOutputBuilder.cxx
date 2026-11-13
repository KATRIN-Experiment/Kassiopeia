#include "KSWriteROOTConditionOutputBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionOutputBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionOutputStructure =
    //! the name of this write condition.
    KSWriteROOTConditionOutputBuilder::Attribute<std::string>("name") +
    //! the writer does only write to disc if the value of the output variable is above this value
    KSWriteROOTConditionOutputBuilder::Attribute<double>("min_value") +
    //! the writer does only write to disc if the value of the output variable is below this value
    KSWriteROOTConditionOutputBuilder::Attribute<double>("max_value") +
    //! the name of the group the output variable is in (optional)
    KSWriteROOTConditionOutputBuilder::Attribute<std::string>("group") +
    //! the name of the output variable which is used for the write decision
    KSWriteROOTConditionOutputBuilder::Attribute<std::string>("parent");


STATICINT sKSWriteROOTConditionOutput =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionOutputData>("kswrite_root_condition_output");

}  // namespace katrin
