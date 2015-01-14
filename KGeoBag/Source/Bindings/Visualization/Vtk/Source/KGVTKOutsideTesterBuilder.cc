#include "KGVTKOutsideTesterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    static const int sKGVTKOutsideTesterStructure =
        KGVTKOutsideTesterBuilder::Attribute< string >( "name" ) +
        KGVTKOutsideTesterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKOutsideTesterBuilder::Attribute< string >( "spaces" ) +
        KGVTKOutsideTesterBuilder::Attribute< KThreeVector >( "sample_disk_origin" ) +
        KGVTKOutsideTesterBuilder::Attribute< KThreeVector >( "sample_disk_normal" ) +
        KGVTKOutsideTesterBuilder::Attribute< double >( "sample_disk_radius" ) +
        KGVTKOutsideTesterBuilder::Attribute< unsigned int >( "sample_count" ) +
        KGVTKOutsideTesterBuilder::Attribute< KGRGBColor >( "inside_color" ) +
        KGVTKOutsideTesterBuilder::Attribute< KGRGBColor >( "outside_color" ) +
        KGVTKOutsideTesterBuilder::Attribute< double >( "vertex_size" );

    static const int sKGVTKOutsideTesterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKOutsideTester >( "vtk_outside_tester" );

}
