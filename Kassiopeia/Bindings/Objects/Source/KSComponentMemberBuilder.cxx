#include "KSComponentMemberBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentBuilder::~KComplexElement() = default;

STATICINT sKSComponentStructure = KSComponentBuilder::Attribute<std::string>("name") +
                                  KSComponentBuilder::Attribute<std::string>("parent") +
                                  KSComponentBuilder::Attribute<std::string>("field");

STATICINT sKSComponent = KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("component_member") +
                         KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("output") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("ks_component_member") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("output");

}  // namespace katrin
