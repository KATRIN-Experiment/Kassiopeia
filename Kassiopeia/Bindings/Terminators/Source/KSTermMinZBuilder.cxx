#include "KSTermMinZBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinZBuilder::~KComplexElement() = default;

STATICINT sKSTermMinZStructure =
    //! the name of the terminator.
    KSTermMinZBuilder::Attribute<std::string>("name") +
    //! the minimum allowed z coordinate.
    KSTermMinZBuilder::Attribute<double>("z");

STATICINT sKSTermMinZ = KSRootBuilder::ComplexElement<KSTermMinZ>("ksterm_min_z");


}  // namespace katrin
