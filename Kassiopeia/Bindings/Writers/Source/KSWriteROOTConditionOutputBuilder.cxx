#include "KSWriteROOTConditionOutputBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionOutputBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionOutputStructure = KSWriteROOTConditionOutputBuilder::Attribute<std::string>("name") +
                                                 KSWriteROOTConditionOutputBuilder::Attribute<double>("min_value") +
                                                 KSWriteROOTConditionOutputBuilder::Attribute<double>("max_value") +
                                                 KSWriteROOTConditionOutputBuilder::Attribute<std::string>("group") +
                                                 KSWriteROOTConditionOutputBuilder::Attribute<std::string>("parent");


STATICINT sKSWriteROOTConditionOutput =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionOutputData>("kswrite_root_condition_output");

}  // namespace katrin
