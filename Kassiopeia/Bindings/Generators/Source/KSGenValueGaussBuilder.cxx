#include "KSGenValueGaussBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueGaussBuilder::~KComplexElement() = default;

STATICINT sKSGenValueGaussStructure =
    //! the name of this value.
    KSGenValueGaussBuilder::Attribute<std::string>("name") +
    //! the minimum value to return.
    KSGenValueGaussBuilder::Attribute<double>("value_min") +
    //! the maximum value to return.
    KSGenValueGaussBuilder::Attribute<double>("value_max") +
    //! the mean of the underlying gaussian distribution.
    KSGenValueGaussBuilder::Attribute<double>("value_mean") +
    //! the square root of the second cumulant of the underlying gaussian distribution.
    KSGenValueGaussBuilder::Attribute<double>("value_sigma");

STATICINT sKSGenValueGauss = KSRootBuilder::ComplexElement<KSGenValueGauss>("ksgen_value_gauss");

}  // namespace katrin
