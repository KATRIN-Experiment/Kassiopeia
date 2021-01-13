#include "KSComponentMaximumBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMaximumBuilder::~KComplexElement() = default;

STATICINT sKSComponentMaximumStructure = KSComponentMaximumBuilder::Attribute<std::string>("name") +
                                         KSComponentMaximumBuilder::Attribute<std::string>("group") +
                                         KSComponentMaximumBuilder::Attribute<std::string>("component") +
                                         KSComponentMaximumBuilder::Attribute<std::string>("parent");

STATICINT sKSComponentMaximum = KSComponentGroupBuilder::ComplexElement<KSComponentMaximumData>("component_maximum") +
                                KSComponentGroupBuilder::ComplexElement<KSComponentMaximumData>("output_maximum") +
                                KSRootBuilder::ComplexElement<KSComponentMaximumData>("ks_component_maximum") +
                                KSRootBuilder::ComplexElement<KSComponentMaximumData>("output_maximum");

}  // namespace katrin
