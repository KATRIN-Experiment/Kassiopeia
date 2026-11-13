#include "KSTrajInterpolatorFastBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajInterpolatorFastBuilder::~KComplexElement() = default;

//! the name of this interpolator.
STATICINT sKSTrajInterpolatorFastStructure = KSTrajInterpolatorFastBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajInterpolatorFast =
    KSRootBuilder::ComplexElement<KSTrajInterpolatorFast>("kstraj_interpolator_fast");

}  // namespace katrin
