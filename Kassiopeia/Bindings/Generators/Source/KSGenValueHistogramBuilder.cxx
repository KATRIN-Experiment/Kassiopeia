#include "KSGenValueHistogramBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueHistogramBuilder::~KComplexElement() = default;

STATICINT sKSGenValueHistogramStructure =
    //! the name of this value.
    KSGenValueHistogramBuilder::Attribute<std::string>("name") +
    //! the filename to load the histogram from.
    KSGenValueHistogramBuilder::Attribute<std::string>("base") +
    //! the path to search for the given file.
    //! uses Kassiopeia's data directory by default.
    KSGenValueHistogramBuilder::Attribute<std::string>("path") +
    //! the histogram from which to draw.
    KSGenValueHistogramBuilder::Attribute<std::string>("histogram") +
    //! (optional) a formula to modify the drawn value.
    KSGenValueHistogramBuilder::Attribute<std::string>("formula");

STATICINT sKSGenValueHistogram = KSRootBuilder::ComplexElement<KSGenValueHistogram>("ksgen_value_histogram");

}  // namespace katrin
