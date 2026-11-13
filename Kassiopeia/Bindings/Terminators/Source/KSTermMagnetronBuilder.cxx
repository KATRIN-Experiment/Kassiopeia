#include "KSTermMagnetronBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

template<> KSTermMagnetronBuilder::~KComplexElement() = default;

STATICINT sKSTermMagnetronStructure =
    //! the name of the terminator
    KSTermMagnetronBuilder::Attribute<std::string>("name") +
    //! the azimuthal angle surpassed under magnetron motion before the particle is killed
    KSTermMagnetronBuilder::Attribute<double>("max_phi");

STATICINT sKSTermMagnetron = KSRootBuilder::ComplexElement<KSTermMagnetron>("ksterm_magnetron");

}  // namespace katrin
