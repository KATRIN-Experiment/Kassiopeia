#include "KSComponentMaximumAtBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMaximumAtBuilder::~KComplexElement() {}

STATICINT sKSComponentMaximumAtStructure =
    KSComponentMaximumAtBuilder::Attribute<string>("name") + KSComponentMaximumAtBuilder::Attribute<string>("group") +
    KSComponentMaximumAtBuilder::Attribute<string>("component") +
    KSComponentMaximumAtBuilder::Attribute<string>("parent") + KSComponentMaximumAtBuilder::Attribute<string>("source");

STATICINT sKSComponentMaximumAt =
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("component_maximum_at") +
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("ks_component_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at");

}  // namespace katrin
