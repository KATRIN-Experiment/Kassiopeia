#include "KSComponentMemberBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentBuilder::~KComplexElement() {}

STATICINT sKSComponentStructure = KSComponentBuilder::Attribute<string>("name") +
                                  KSComponentBuilder::Attribute<string>("parent") +
                                  KSComponentBuilder::Attribute<string>("field");

STATICINT sKSComponent = KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("component_member") +
                         KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("output") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("ks_component_member") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("output");

}  // namespace katrin
