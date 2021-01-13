#include "KSComponentMinimumBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMinimumBuilder::~KComplexElement() = default;

STATICINT sKSComponentMinimumStructure = KSComponentMinimumBuilder::Attribute<std::string>("name") +
                                         KSComponentMinimumBuilder::Attribute<std::string>("group") +
                                         KSComponentMinimumBuilder::Attribute<std::string>("component") +
                                         KSComponentMinimumBuilder::Attribute<std::string>("parent");

STATICINT sKSComponentMinimum = KSComponentGroupBuilder::ComplexElement<KSComponentMinimumData>("component_minimum") +
                                KSComponentGroupBuilder::ComplexElement<KSComponentMinimumData>("output_minimum") +
                                KSRootBuilder::ComplexElement<KSComponentMinimumData>("ks_component_minimum") +
                                KSRootBuilder::ComplexElement<KSComponentMinimumData>("output_minimum");

}  // namespace katrin
