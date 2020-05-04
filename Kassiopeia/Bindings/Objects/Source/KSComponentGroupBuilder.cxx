#include "KSComponentGroupBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentGroupBuilder::~KComplexElement() {}

STATICINT sKSGroupStructure = KSComponentGroupBuilder::Attribute<string>("name");

STATICINT sKSGroup = KSRootBuilder::ComplexElement<KSComponentGroup>("ks_component_group") +
                     KSRootBuilder::ComplexElement<KSComponentGroup>("output_group");

}  // namespace katrin
