#include "KGVTKRandomPointTesterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    STATICINT sKGVTKRandomPointTesterStructure =
		KGVTKRandomPointTesterBuilder::Attribute< string >( "name" ) +
		KGVTKRandomPointTesterBuilder::Attribute< string >( "surfaces" ) +
		KGVTKRandomPointTesterBuilder::Attribute< string >( "spaces" ) +
		KGVTKRandomPointTesterBuilder::Attribute< KGRGBColor >( "sample_color" ) +
		KGVTKRandomPointTesterBuilder::Attribute< double >( "vertex_size" );

    STATICINT sKGVTKRandomPointTesterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKRandomPointTester >( "vtk_random_point_tester" );

}
