#include "KSComponentMathBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMathBuilder::~KComplexElement() = default;

STATICINT sKSComponentMathStructure =
    KSComponentMathBuilder::Attribute<std::string>("name") + KSComponentMathBuilder::Attribute<std::string>("group") +
    KSComponentMathBuilder::Attribute<std::string>("component") +
    KSComponentMathBuilder::Attribute<std::string>("parent") + KSComponentMathBuilder::Attribute<std::string>("term");

STATICINT sKSComponentMath = KSComponentGroupBuilder::ComplexElement<KSComponentMathData>("component_math") +
                             KSComponentGroupBuilder::ComplexElement<KSComponentMathData>("output_math") +
                             KSRootBuilder::ComplexElement<KSComponentMathData>("ks_component_math") +
                             KSRootBuilder::ComplexElement<KSComponentMathData>("output_math");

}  // namespace katrin
