#include "KSComponentMinimumAtBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMinimumAtBuilder::~KComplexElement() {}

STATICINT sKSComponentMinimumAtStructure =
    KSComponentMinimumAtBuilder::Attribute<string>("name") + KSComponentMinimumAtBuilder::Attribute<string>("group") +
    KSComponentMinimumAtBuilder::Attribute<string>("component") +
    KSComponentMinimumAtBuilder::Attribute<string>("parent") + KSComponentMinimumAtBuilder::Attribute<string>("source");

STATICINT sKSComponentMinimumAt =
    KSComponentGroupBuilder::ComplexElement<KSComponentMinimumAtData>("component_minimum_at") +
    KSComponentGroupBuilder::ComplexElement<KSComponentMinimumAtData>("output_minimum_at") +
    KSRootBuilder::ComplexElement<KSComponentMinimumAtData>("ks_component_minimum_at") +
    KSRootBuilder::ComplexElement<KSComponentMinimumAtData>("output_minimum_at");

}  // namespace katrin
