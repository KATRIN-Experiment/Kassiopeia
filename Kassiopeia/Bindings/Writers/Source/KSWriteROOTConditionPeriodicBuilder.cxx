#include "KSWriteROOTConditionPeriodicBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionPeriodicBuilder::~KComplexElement() {}

STATICINT sKSWriteROOTConditionPeriodicStructure =
    KSWriteROOTConditionPeriodicBuilder::Attribute<string>("name") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("initial_min") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("initial_max") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("increment") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("reset_min") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<double>("reset_max") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<string>("group") +
    KSWriteROOTConditionPeriodicBuilder::Attribute<string>("parent");


STATICINT sKSWriteROOTConditionPeriodic =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionPeriodicData>("kswrite_root_condition_periodic");

}  // namespace katrin
