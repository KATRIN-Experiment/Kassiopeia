#include "KSGeoSideBuilder.h"

#include "KSCommandMemberBuilder.h"
#include "KSCommandMemberSimpleBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

STATICINT sKSGeoSideStructure =
    KSGeoSideBuilder::Attribute<string>("name") + KSGeoSideBuilder::Attribute<string>("surfaces") +
    KSGeoSideBuilder::Attribute<string>("spaces") + KSGeoSideBuilder::ComplexElement<KSCommandMemberData>("command") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddTerminatorData>("add_terminator") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveTerminatorData>("remove_terminator") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddMagneticFieldData>("add_magnetic_field") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveMagneticFieldData>("remove_magnetic_field") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddElectricFieldData>("add_electric_field") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveElectricFieldData>("remove_magnetic_field") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberSetSurfaceInteractionData>("set_surface_interaction") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberClearSurfaceInteractionData>("clear_surface_interaction") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddStepModifierData>("add_step_modifier") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveStepModifierData>("remove_step_modifier") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddStepOutputData>("add_step_output") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveStepOutputData>("remove_step_output") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberAddTrackOutputData>("add_track_output") +
    KSGeoSideBuilder::ComplexElement<KSCommandMemberRemoveTrackOutputData>("remove_track_output");

STATICINT sKSGeoSide = KSRootBuilder::ComplexElement<KSGeoSide>("ksgeo_side");

}  // namespace katrin
