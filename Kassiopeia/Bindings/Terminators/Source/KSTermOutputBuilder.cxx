#include "KSTermOutputBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermOutputBuilder::~KComplexElement() = default;

STATICINT sKSTermOutputStructure =
    KSTermOutputBuilder::Attribute<std::string>("name") + KSTermOutputBuilder::Attribute<double>("min_value") +
    KSTermOutputBuilder::Attribute<double>("max_value") + KSTermOutputBuilder::Attribute<std::string>("group") +
    KSTermOutputBuilder::Attribute<std::string>("component");


STATICINT sKSTermOutput = KSRootBuilder::ComplexElement<KSTermOutputData>("ksterm_output");

}  // namespace katrin
