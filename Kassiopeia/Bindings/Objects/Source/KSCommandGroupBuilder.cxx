#include "KSCommandGroupBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSCommandGroupBuilder::~KComplexElement() = default;

STATICINT sKSGroupStructure =
    KSCommandGroupBuilder::Attribute<std::string>("name") + KSCommandGroupBuilder::Attribute<std::string>("command");

STATICINT sKSGroup = KSCommandGroupBuilder::ComplexElement<KSCommandGroup>("command_group") +
                     KSRootBuilder::ComplexElement<KSCommandGroup>("ks_command_group");

}  // namespace katrin
