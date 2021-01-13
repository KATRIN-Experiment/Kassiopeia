#include "KSTrajControlSpinPrecessionBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlSpinPrecessionBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlSpinPrecessionStructure = KSTrajControlSpinPrecessionBuilder::Attribute<std::string>("name") +
                                                  KSTrajControlSpinPrecessionBuilder::Attribute<double>("fraction");

STATICINT sToolboxKSTrajControlSpinPrecession =
    KSRootBuilder::ComplexElement<KSTrajControlSpinPrecession>("kstraj_control_spin_precession");

}  // namespace katrin
