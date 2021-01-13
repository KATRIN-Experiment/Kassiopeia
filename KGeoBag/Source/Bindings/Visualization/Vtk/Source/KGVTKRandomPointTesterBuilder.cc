#include "KGVTKRandomPointTesterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKRandomPointTesterStructure = KGVTKRandomPointTesterBuilder::Attribute<std::string>("name") +
                                             KGVTKRandomPointTesterBuilder::Attribute<std::string>("surfaces") +
                                             KGVTKRandomPointTesterBuilder::Attribute<std::string>("spaces") +
                                             KGVTKRandomPointTesterBuilder::Attribute<KGRGBColor>("sample_color") +
                                             KGVTKRandomPointTesterBuilder::Attribute<double>("vertex_size");

STATICINT sKGVTKRandomPointTesterWindow =
    KVTKWindowBuilder::ComplexElement<KGVTKRandomPointTester>("vtk_random_point_tester");

}  // namespace katrin
