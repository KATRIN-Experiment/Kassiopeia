#include "KSWriteROOTConditionStepBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionStepBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionStepStructure = KSWriteROOTConditionStepBuilder::Attribute<std::string>("name") +
                                               KSWriteROOTConditionStepBuilder::Attribute<int>("nth_step") +
                                               KSWriteROOTConditionStepBuilder::Attribute<std::string>("group") +
                                               KSWriteROOTConditionStepBuilder::Attribute<std::string>("parent");


STATICINT sKSWriteROOTConditionStep =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionStepData>("kswrite_root_condition_step");

}  // namespace katrin
