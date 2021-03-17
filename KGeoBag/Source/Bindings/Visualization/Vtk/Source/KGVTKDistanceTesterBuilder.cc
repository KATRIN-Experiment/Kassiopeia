#include "KGVTKDistanceTesterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKDistanceTesterStructure = KGVTKDistanceTesterBuilder::Attribute<std::string>("name") +
                                          KGVTKDistanceTesterBuilder::Attribute<std::string>("surfaces") +
                                          KGVTKDistanceTesterBuilder::Attribute<std::string>("spaces") +
                                          KGVTKDistanceTesterBuilder::Attribute<KThreeVector>("sample_disk_origin") +
                                          KGVTKDistanceTesterBuilder::Attribute<KThreeVector>("sample_disk_normal") +
                                          KGVTKDistanceTesterBuilder::Attribute<double>("sample_disk_radius") +
                                          KGVTKDistanceTesterBuilder::Attribute<unsigned int>("sample_count") +
                                          KGVTKDistanceTesterBuilder::Attribute<double>("vertex_size");

STATICINT sKGVTKDistanceTesterWindow = KVTKWindowBuilder::ComplexElement<KGVTKDistanceTester>("vtk_distance_tester");

}  // namespace katrin
