#include "KSComponentDeltaBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentDeltaBuilder::~KComplexElement() = default;

STATICINT sKSComponentDeltaStructure = KSComponentDeltaBuilder::Attribute<std::string>("name") +
                                       KSComponentDeltaBuilder::Attribute<std::string>("group") +
                                       KSComponentDeltaBuilder::Attribute<std::string>("component") +
                                       KSComponentDeltaBuilder::Attribute<std::string>("parent");


STATICINT sKSComponentDelta = KSComponentGroupBuilder::ComplexElement<KSComponentDeltaData>("component_delta") +
                              KSComponentGroupBuilder::ComplexElement<KSComponentDeltaData>("output_delta") +
                              KSRootBuilder::ComplexElement<KSComponentDeltaData>("ks_component_delta") +
                              KSRootBuilder::ComplexElement<KSComponentDeltaData>("output_delta");

}  // namespace katrin
