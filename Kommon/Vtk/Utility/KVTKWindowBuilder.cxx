//
// Created by trost on 25.07.16.
//

#include "KRoot.h"
#include "KVTKWindowBuilder.h"
#include "KElementProcessor.hh"

using namespace std;

namespace katrin
{

STATICINT sKVTKWindow =
    KRootBuilder::ComplexElement< KVTKWindow >( "vtk_window" );

STATICINT sKVTKWindowCompat =
    KElementProcessor::ComplexElement< KVTKWindow >( "vtk_window" );

STATICINT sKVTKWindowStructure =
        KVTKWindowBuilder::Attribute< string >( "name" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_display" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_write" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_help" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_axis" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_data" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_parallel_projection" ) +
        KVTKWindowBuilder::Attribute< string >( "frame_title" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "frame_size_x" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "frame_size_y" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_red" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_green" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_blue" ) +
        KVTKWindowBuilder::Attribute< double >( "eye_angle" ) +
        KVTKWindowBuilder::Attribute< double >( "view_angle" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "multi_samples" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "depth_peeling" );

}
