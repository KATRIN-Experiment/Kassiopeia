#include "KGVTKPointTesterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    STATICINT sKGVTKPointTesterStructure =
        KGVTKPointTesterBuilder::Attribute< string >( "name" ) +
        KGVTKPointTesterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKPointTesterBuilder::Attribute< string >( "spaces" ) +
        KGVTKPointTesterBuilder::Attribute< KThreeVector >( "sample_disk_origin" ) +
        KGVTKPointTesterBuilder::Attribute< KThreeVector >( "sample_disk_normal" ) +
        KGVTKPointTesterBuilder::Attribute< double >( "sample_disk_radius" ) +
        KGVTKPointTesterBuilder::Attribute< unsigned int >( "sample_count" ) +
        KGVTKPointTesterBuilder::Attribute< KGRGBColor >( "sample_color" ) +
        KGVTKPointTesterBuilder::Attribute< KGRGBColor >( "point_color" ) +
        KGVTKPointTesterBuilder::Attribute< double >( "vertex_size" ) +
        KGVTKPointTesterBuilder::Attribute< double >( "line_size" );

    STATICINT sKGVTKPointTesterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKPointTester >( "vtk_point_tester" );

}
