#include "KSComponentMaximumAtBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMaximumAtBuilder::~KComplexElement() = default;

STATICINT sKSComponentMaximumAtStructure = KSComponentMaximumAtBuilder::Attribute<std::string>("name") +
                                           KSComponentMaximumAtBuilder::Attribute<std::string>("group") +
                                           KSComponentMaximumAtBuilder::Attribute<std::string>("component") +
                                           KSComponentMaximumAtBuilder::Attribute<std::string>("parent") +
                                           KSComponentMaximumAtBuilder::Attribute<std::string>("source");

STATICINT sKSComponentMaximumAt =
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("component_maximum_at") +
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("ks_component_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at");

}  // namespace katrin
