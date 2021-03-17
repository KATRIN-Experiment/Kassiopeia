#include "KSGeoSpaceBuilder.h"

#include "KSCommandMemberBuilder.h"
#include "KSCommandMemberSimpleBuilder.h"
#include "KSGeoSideBuilder.h"
#include "KSGeoSurfaceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

STATICINT sKSSpaceStructure =
    KSGeoSpaceBuilder::Attribute<std::string>("name") + KSGeoSpaceBuilder::Attribute<std::string>("spaces") +
    KSGeoSpaceBuilder::ComplexElement<KSGeoSpace>("geo_space") +
    KSGeoSpaceBuilder::ComplexElement<KSGeoSurface>("geo_surface") +
    KSGeoSpaceBuilder::ComplexElement<KSGeoSide>("geo_side") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberData>("command") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddTerminatorData>("add_terminator") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveTerminatorData>("remove_terminator") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddMagneticFieldData>("add_magnetic_field") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveMagneticFieldData>("remove_magnetic_field") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddElectricFieldData>("add_electric_field") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveElectricFieldData>("remove_electric_field") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddSpaceInteractionData>("add_space_interaction") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveSpaceInteractionData>("remove_space_interaction") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetDensityData>("set_density") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearDensityData>("clear_density") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddStepModifierData>("add_step_modifier") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveStepModifierData>("remove_step_modifier") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetTrajectoryData>("set_trajectory") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearTrajectoryData>("clear_trajectory") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddControlData>("add_control") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveControlData>("remove_control") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddTermData>("add_term") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveTermData>("remove_term") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddStepOutputData>("add_step_output") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveStepOutputData>("remove_step_output") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberAddTrackOutputData>("add_track_output") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberRemoveTrackOutputData>("remove_track_output") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetStepPointData>("set_step_point") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearStepPointData>("clear_step_point") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetStepDataData>("set_step_data") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearStepDataData>("clear_step_data") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetTrackPointData>("set_track_point") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearTrackPointData>("clear_track_point") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberSetTrackDataData>("set_track_data") +
    KSGeoSpaceBuilder::ComplexElement<KSCommandMemberClearTrackDataData>("clear_track_data");


STATICINT sKSSpace = KSRootBuilder::ComplexElement<KSGeoSpace>("ksgeo_space");

}  // namespace katrin
