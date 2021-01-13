#include "KSComponentMinimumAtBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMinimumAtBuilder::~KComplexElement() = default;

STATICINT sKSComponentMinimumAtStructure = KSComponentMinimumAtBuilder::Attribute<std::string>("name") +
                                           KSComponentMinimumAtBuilder::Attribute<std::string>("group") +
                                           KSComponentMinimumAtBuilder::Attribute<std::string>("component") +
                                           KSComponentMinimumAtBuilder::Attribute<std::string>("parent") +
                                           KSComponentMinimumAtBuilder::Attribute<std::string>("source");

STATICINT sKSComponentMinimumAt =
    KSComponentGroupBuilder::ComplexElement<KSComponentMinimumAtData>("component_minimum_at") +
    KSComponentGroupBuilder::ComplexElement<KSComponentMinimumAtData>("output_minimum_at") +
    KSRootBuilder::ComplexElement<KSComponentMinimumAtData>("ks_component_minimum_at") +
    KSRootBuilder::ComplexElement<KSComponentMinimumAtData>("output_minimum_at");

}  // namespace katrin
