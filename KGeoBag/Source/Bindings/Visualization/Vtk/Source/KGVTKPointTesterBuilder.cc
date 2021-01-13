#include "KGVTKPointTesterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKPointTesterStructure = KGVTKPointTesterBuilder::Attribute<std::string>("name") +
                                       KGVTKPointTesterBuilder::Attribute<std::string>("surfaces") +
                                       KGVTKPointTesterBuilder::Attribute<std::string>("spaces") +
                                       KGVTKPointTesterBuilder::Attribute<KThreeVector>("sample_disk_origin") +
                                       KGVTKPointTesterBuilder::Attribute<KThreeVector>("sample_disk_normal") +
                                       KGVTKPointTesterBuilder::Attribute<double>("sample_disk_radius") +
                                       KGVTKPointTesterBuilder::Attribute<unsigned int>("sample_count") +
                                       KGVTKPointTesterBuilder::Attribute<KGRGBColor>("sample_color") +
                                       KGVTKPointTesterBuilder::Attribute<KGRGBColor>("point_color") +
                                       KGVTKPointTesterBuilder::Attribute<double>("vertex_size") +
                                       KGVTKPointTesterBuilder::Attribute<double>("line_size");

STATICINT sKGVTKPointTesterWindow = KVTKWindowBuilder::ComplexElement<KGVTKPointTester>("vtk_point_tester");

}  // namespace katrin
