#include "KSNavSurfaceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSNavSurfaceBuilder::~KComplexElement() = default;

STATICINT sKSNavSurfaceStructure = KSNavSurfaceBuilder::Attribute<std::string>("name") +
                                   KSNavSurfaceBuilder::Attribute<bool>("transmission_split") +
                                   KSNavSurfaceBuilder::Attribute<bool>("reflection_split");

STATICINT sToolboxKSNavSurface = KSRootBuilder::ComplexElement<KSNavSurface>("ksnav_surface");


}  // namespace katrin
