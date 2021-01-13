//
// Created by trost on 07.03.16.
//

#include "KSWriteROOTConditionTerminatorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionTerminatorBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTConditionTerminatorStructure =
    KSWriteROOTConditionTerminatorBuilder::Attribute<std::string>("name") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<std::string>("group") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<std::string>("parent") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<std::string>("match_terminator");


STATICINT sKSWriteROOTConditionTerminator =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionTerminatorData>("kswrite_root_condition_terminator");

}  // namespace katrin