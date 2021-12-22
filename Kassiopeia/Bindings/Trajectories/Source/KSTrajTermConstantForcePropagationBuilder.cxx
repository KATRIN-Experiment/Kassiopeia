#include "KSTrajTermConstantForcePropagationBuilder.h"

#include "KSRootBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermConstantForcePropagationBuilder::~KComplexElement() = default;

STATICINT sKSTrajTermConstantForcePropagationStructure =
    KSTrajTermConstantForcePropagationBuilder::Attribute<std::string>("name") +
    KSTrajTermConstantForcePropagationBuilder::Attribute<KThreeVector>("force");

STATICINT sToolboxKSTrajTermConstantForcePropagation =
    KSRootBuilder::ComplexElement<KSTrajTermConstantForcePropagation>("kstraj_term_constant_force_propagation");

}  // namespace katrin
