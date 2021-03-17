#include "KSTrajInterpolatorHermiteBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajInterpolatorHermiteBuilder::~KComplexElement() = default;

STATICINT sKSTrajInterpolatorHermiteStructure = KSTrajInterpolatorHermiteBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajInterpolatorHermite =
    KSRootBuilder::ComplexElement<KSTrajInterpolatorHermite>("kstraj_interpolator_hermite");

}  // namespace katrin
