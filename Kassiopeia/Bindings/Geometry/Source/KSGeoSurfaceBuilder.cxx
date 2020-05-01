#include "KSGeoSurfaceBuilder.h"

#include "KSCommandMemberBuilder.h"
#include "KSCommandMemberSimpleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

STATICINT sKSGeoSurfaceStructure =
    KSGeoSurfaceBuilder::Attribute<string>("name") + KSGeoSurfaceBuilder::Attribute<string>("surfaces") +
    KSGeoSurfaceBuilder::Attribute<string>("spaces") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberData>("command") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddTerminatorData>("add_terminator") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveTerminatorData>("remove_terminator") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddMagneticFieldData>("add_magnetic_field") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveMagneticFieldData>("remove_magnetic_field") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddElectricFieldData>("add_electric_field") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveElectricFieldData>("remove_magnetic_field") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberSetSurfaceInteractionData>("set_surface_interaction") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberClearSurfaceInteractionData>("clear_surface_interaction") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddStepModifierData>("add_step_modifier") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveStepModifierData>("remove_step_modifier") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddStepOutputData>("add_step_output") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveStepOutputData>("remove_step_output") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberAddTrackOutputData>("add_track_output") +
    KSGeoSurfaceBuilder::ComplexElement<KSCommandMemberRemoveTrackOutputData>("remove_track_output");

STATICINT sKSGeoSurface = KSRootBuilder::ComplexElement<KSGeoSurface>("ksgeo_surface");

}  // namespace katrin
