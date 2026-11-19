#include "KSTrajControlLengthBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlLengthBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlLengthStructure =
    //! the name of this control.
    KSTrajControlLengthBuilder::Attribute<std::string>("name") +
    //! the length in meters from which to calculate the temporal step.
    KSTrajControlLengthBuilder::Attribute<double>("length");

STATICINT sToolboxKSTrajControlLength = KSRootBuilder::ComplexElement<KSTrajControlLength>("kstraj_control_length");

}  // namespace katrin
