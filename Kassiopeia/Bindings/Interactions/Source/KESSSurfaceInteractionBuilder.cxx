#include "KESSSurfaceInteractionBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KESSSurfaceInteractionBuilder::~KComplexElement() = default;

STATICINT sKESSSSurfaceInteractionStructure = KESSSurfaceInteractionBuilder::Attribute<std::string>("name") +
                                              KESSSurfaceInteractionBuilder::Attribute<std::string>("siliconside");

STATICINT sKESSSSurfaceInteractionElement =
    KSRootBuilder::ComplexElement<KESSSurfaceInteraction>("kess_surface_interaction");
}  // namespace katrin
