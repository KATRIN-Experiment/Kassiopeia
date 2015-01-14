#include "KGPlanarPolyLineBuilder.hh"

namespace katrin
{

    static const int sKGPlanarPolyLineStartPointArgumentsBuilderStructure =
        KGPlanarPolyLineStartPointArgumentsBuilder::Attribute< double >( "x" ) +
        KGPlanarPolyLineStartPointArgumentsBuilder::Attribute< double >( "y" );

    static const int sKGPlanarPolyLineLineArgumentsBuilderStructure =
        KGPlanarPolyLineLineArgumentsBuilder::Attribute< double >( "x" ) +
        KGPlanarPolyLineLineArgumentsBuilder::Attribute< double >( "y" ) +
        KGPlanarPolyLineLineArgumentsBuilder::Attribute< unsigned int >( "line_mesh_count" ) +
        KGPlanarPolyLineLineArgumentsBuilder::Attribute< double >( "line_mesh_power" );

    static const int sKGPlanarPolyLineArcArgumentsBuilderStructure =
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< double >( "x" ) +
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< double >( "y" ) +
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< double >( "radius" ) +
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< bool >( "right" ) +
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< bool >( "short" ) +
        KGPlanarPolyLineArcArgumentsBuilder::Attribute< unsigned int >( "arc_mesh_count" );

    static const int sKGPlanarPolyLineBuilderStructure =
        KGPlanarPolyLineBuilder::ComplexElement< KGPlanarPolyLine::StartPointArguments >( "start_point" ) +
        KGPlanarPolyLineBuilder::ComplexElement< KGPlanarPolyLine::LineArguments >( "next_line" ) +
        KGPlanarPolyLineBuilder::ComplexElement< KGPlanarPolyLine::ArcArguments >( "next_arc" ) +
        KGPlanarPolyLineBuilder::ComplexElement< KGPlanarPolyLine::LineArguments >( "previous_line" ) +
        KGPlanarPolyLineBuilder::ComplexElement< KGPlanarPolyLine::ArcArguments >( "previous_arc" );

}
