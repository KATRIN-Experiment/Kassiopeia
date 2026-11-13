#include "KSTermOutputBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermOutputBuilder::~KComplexElement() = default;

STATICINT sKSTermOutputStructure =
    //! the name of the terminator
    KSTermOutputBuilder::Attribute<std::string>("name") +
    //! the minimal value
    //!
    //! max_value
    //! the maximal value
    KSTermOutputBuilder::Attribute<double>("min_value") + KSTermOutputBuilder::Attribute<double>("max_value") +
    //! the output group with the specific output component
    KSTermOutputBuilder::Attribute<std::string>("group") +
    //! the output component (must be of int or double like typ)
    KSTermOutputBuilder::Attribute<std::string>("component");


STATICINT sKSTermOutput = KSRootBuilder::ComplexElement<KSTermOutputData>("ksterm_output");

}  // namespace katrin
