#include "KSRootSpaceInteractionBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootSpaceInteractionBuilder::~KComplexElement() {}

STATICINT sKSRootSpaceInteraction = KSRootBuilder::ComplexElement<KSRootSpaceInteraction>("ks_root_space_interaction");

STATICINT sKSRootSpaceInteractionStructure = KSRootSpaceInteractionBuilder::Attribute<string>("name") +
                                             KSRootSpaceInteractionBuilder::Attribute<string>("add_space_interaction");

}  // namespace katrin
