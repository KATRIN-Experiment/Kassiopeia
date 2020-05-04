#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

STATICINT sKGPlanarPolyLoopStartPointArgumentsBuilderStructure =
    KGPlanarPolyLoopStartPointArgumentsBuilder::Attribute<double>("x") +
    KGPlanarPolyLoopStartPointArgumentsBuilder::Attribute<double>("y");

STATICINT sKGPlanarPolyLoopLineArgumentsBuilderStructure =
    KGPlanarPolyLoopLineArgumentsBuilder::Attribute<double>("x") +
    KGPlanarPolyLoopLineArgumentsBuilder::Attribute<double>("y") +
    KGPlanarPolyLoopLineArgumentsBuilder::Attribute<unsigned int>("line_mesh_count") +
    KGPlanarPolyLoopLineArgumentsBuilder::Attribute<double>("line_mesh_power");

STATICINT sKGPlanarPolyLoopArcArgumentsBuilderStructure =
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<double>("x") +
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<double>("y") +
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<double>("radius") +
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<bool>("right") +
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<bool>("short") +
    KGPlanarPolyLoopArcArgumentsBuilder::Attribute<unsigned int>("arc_mesh_count");

STATICINT sKGPlanarPolyLoopLastLineArgumentsBuilderStructure =
    KGPlanarPolyLoopLastLineArgumentsBuilder::Attribute<unsigned int>("line_mesh_count") +
    KGPlanarPolyLoopLastLineArgumentsBuilder::Attribute<double>("line_mesh_power");

STATICINT sKGPlanarPolyLoopLastArcArgumentsBuilderStructure =
    KGPlanarPolyLoopLastArcArgumentsBuilder::Attribute<double>("radius") +
    KGPlanarPolyLoopLastArcArgumentsBuilder::Attribute<bool>("right") +
    KGPlanarPolyLoopLastArcArgumentsBuilder::Attribute<bool>("short") +
    KGPlanarPolyLoopLastArcArgumentsBuilder::Attribute<unsigned int>("arc_mesh_count");

STATICINT sKGPlanarPolyLoopBuilderStructure =
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::StartPointArguments>("start_point") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::LineArguments>("next_line") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::ArcArguments>("next_arc") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::LineArguments>("previous_line") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::ArcArguments>("previous_arc") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::LastLineArguments>("last_line") +
    KGPlanarPolyLoopBuilder::ComplexElement<KGPlanarPolyLoop::LastArcArguments>("last_arc");

}  // namespace katrin
