#include "KSTermStepsizeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermStepsizeBuilder::~KComplexElement() = default;

STATICINT sKSTermMinEnergyStructure = KSTermStepsizeBuilder::Attribute<std::string>("name") +
                                      KSTermStepsizeBuilder::Attribute<double>("min_length") +
                                      KSTermStepsizeBuilder::Attribute<double>("max_length");

STATICINT sKSTermMinEnergy = KSRootBuilder::ComplexElement<KSTermStepsize>("ksterm_stepsize");

}  // namespace katrin
