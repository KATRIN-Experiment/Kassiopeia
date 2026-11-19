#include "KSCommandMemberBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSCommandMemberBuilder::~KComplexElement() = default;

STATICINT sKSCommandStructure =
    //! the name of the resulting command object.
    KSCommandMemberBuilder::Attribute<std::string>("name") +
    //! the name of the parent object.
    KSCommandMemberBuilder::Attribute<std::string>("parent") +
    //! the name of the child object to bind with the member.
    KSCommandMemberBuilder::Attribute<std::string>("child") +
    //! the name of the parent class member to use.
    //! available members depend completely on the parent's final type.
    KSCommandMemberBuilder::Attribute<std::string>("field");

STATICINT sKSCommand = KSRootBuilder::ComplexElement<KSCommandMemberData>("ks_command_member");

}  // namespace katrin
