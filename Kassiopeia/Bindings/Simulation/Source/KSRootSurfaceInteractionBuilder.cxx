#include "KSRootSurfaceInteractionBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootSurfaceInteractionBuilder::~KComplexElement() = default;

STATICINT sKSRootSurfaceInteraction =
    KSRootBuilder::ComplexElement<KSRootSurfaceInteraction>("ks_root_surface_interaction");

STATICINT sKSRootSurfaceInteractionStructure =
    KSRootSurfaceInteractionBuilder::Attribute<std::string>("name") +
    KSRootSurfaceInteractionBuilder::Attribute<std::string>("set_surface_interaction");

}  // namespace katrin
