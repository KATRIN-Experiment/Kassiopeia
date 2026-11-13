#include "KSComponentMemberBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentBuilder::~KComplexElement() = default;

STATICINT sKSComponentStructure =
    //! the name of the extracted component.
    KSComponentBuilder::Attribute<std::string>("name") +
    //! the name of the composite.
    KSComponentBuilder::Attribute<std::string>("parent") +
    //! the name of the class member to use.
    //! available members depend completely on the parent's final type.
    KSComponentBuilder::Attribute<std::string>("field");

STATICINT sKSComponent = KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("component_member") +
                         //! adds a component member to the group.
                         KSComponentGroupBuilder::ComplexElement<KSComponentMemberData>("output") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("ks_component_member") +
                         KSRootBuilder::ComplexElement<KSComponentMemberData>("output");

}  // namespace katrin
