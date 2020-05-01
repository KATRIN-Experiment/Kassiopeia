//
// Created by trost on 07.03.16.
//

#include "KSWriteROOTConditionTerminatorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTConditionTerminatorBuilder::~KComplexElement() {}

STATICINT sKSWriteROOTConditionTerminatorStructure =
    KSWriteROOTConditionTerminatorBuilder::Attribute<string>("name") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<string>("group") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<string>("parent") +
    KSWriteROOTConditionTerminatorBuilder::Attribute<string>("match_terminator");


STATICINT sKSWriteROOTConditionTerminator =
    KSRootBuilder::ComplexElement<KSWriteROOTConditionTerminatorData>("kswrite_root_condition_terminator");

}  // namespace katrin