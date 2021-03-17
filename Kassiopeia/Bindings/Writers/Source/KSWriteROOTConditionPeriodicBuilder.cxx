#include "KSWriteROOTConditionPeriodicBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionPeriodicBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionPeriodicStructure =
    KSWriteROOTConditionPeriodicBuilder::Attribute<std::string>("name") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("initial_min") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("initial_max") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("increment") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("reset_min") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("reset_max") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<std::string>("group") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<std::string>("parent");


STATICINT sKSWriteROOTConditionPeriodic =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionPeriodicData>("kswrite_root_condition_periodic");

}  // namespace katrin
