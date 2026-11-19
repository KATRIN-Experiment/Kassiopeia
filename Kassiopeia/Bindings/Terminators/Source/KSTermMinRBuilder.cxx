#include "KSTermMinRBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinRBuilder::~KComplexElement() = default;

STATICINT sKSTermMinRStructure =
    //! the name of the terminator.
    KSTermMinRBuilder::Attribute<std::string>("name") +
    //! the minimum allowed radial coordinate.
    KSTermMinRBuilder::Attribute<double>("r");

STATICINT sKSTermMinR = KSRootBuilder::ComplexElement<KSTermMinR>("ksterm_min_r");

}  // namespace katrin
