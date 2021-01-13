#include "KSCommandMemberBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSCommandMemberBuilder::~KComplexElement() = default;

STATICINT sKSCommandStructure =
    KSCommandMemberBuilder::Attribute<std::string>("name") + KSCommandMemberBuilder::Attribute<std::string>("parent") +
    KSCommandMemberBuilder::Attribute<std::string>("child") + KSCommandMemberBuilder::Attribute<std::string>("field");

STATICINT sKSCommand = KSRootBuilder::ComplexElement<KSCommandMemberData>("ks_command_member");

}  // namespace katrin
