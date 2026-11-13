#include "KSTrajControlTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlTimeBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlTimeStructure =
    //! the name of this control.
    KSTrajControlTimeBuilder::Attribute<std::string>("name") +
    //! the temporal step to use in seconds.
    KSTrajControlTimeBuilder::Attribute<double>("time");

STATICINT sToolboxKSTrajControlTime = KSRootBuilder::ComplexElement<KSTrajControlTime>("kstraj_control_time");

}  // namespace katrin
