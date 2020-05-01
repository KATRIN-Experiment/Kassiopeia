#include "KSGenValueHistogramBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueHistogramBuilder::~KComplexElement() {}

STATICINT sKSGenValueHistogramStructure =
    KSGenValueHistogramBuilder::Attribute<string>("name") + KSGenValueHistogramBuilder::Attribute<string>("base") +
    KSGenValueHistogramBuilder::Attribute<string>("path") + KSGenValueHistogramBuilder::Attribute<string>("histogram") +
    KSGenValueHistogramBuilder::Attribute<string>("formula");

STATICINT sKSGenValueHistogram = KSRootBuilder::ComplexElement<KSGenValueHistogram>("ksgen_value_histogram");

}  // namespace katrin
