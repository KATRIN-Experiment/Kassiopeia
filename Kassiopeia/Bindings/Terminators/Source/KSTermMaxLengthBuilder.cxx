#include "KSTermMaxLengthBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxLengthBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxLengthStructure =
    //! the name of the terminator.
    KSTermMaxLengthBuilder::Attribute<std::string>("name") +
    //! the maximum allowed track length.
    KSTermMaxLengthBuilder::Attribute<double>("length");

STATICINT sKSTermMaxLength = KSRootBuilder::ComplexElement<KSTermMaxLength>("ksterm_max_length");

}  // namespace katrin
