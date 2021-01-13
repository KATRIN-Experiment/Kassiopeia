#include "KSGenValueHistogramBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueHistogramBuilder::~KComplexElement() = default;

STATICINT sKSGenValueHistogramStructure = KSGenValueHistogramBuilder::Attribute<std::string>("name") +
                                          KSGenValueHistogramBuilder::Attribute<std::string>("base") +
                                          KSGenValueHistogramBuilder::Attribute<std::string>("path") +
                                          KSGenValueHistogramBuilder::Attribute<std::string>("histogram") +
                                          KSGenValueHistogramBuilder::Attribute<std::string>("formula");

STATICINT sKSGenValueHistogram = KSRootBuilder::ComplexElement<KSGenValueHistogram>("ksgen_value_histogram");

}  // namespace katrin
