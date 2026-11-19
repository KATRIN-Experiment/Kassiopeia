#include "KSNavSurfaceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSNavSurfaceBuilder::~KComplexElement() = default;

STATICINT sKSNavSurfaceStructure =
    //! the name of the navigator.
    KSNavSurfaceBuilder::Attribute<std::string>("name") +
    //! if true, the navigator will split tracks when particles are transmitted through surfaces.
    KSNavSurfaceBuilder::Attribute<bool>("transmission_split") +
    //! if true, the navigator will split tracks when particles are reflected from surfaces.
    KSNavSurfaceBuilder::Attribute<bool>("reflection_split");

STATICINT sToolboxKSNavSurface = KSRootBuilder::ComplexElement<KSNavSurface>("ksnav_surface");


}  // namespace katrin
