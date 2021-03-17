#ifndef Kassiopeia_KSCommandMemberSimpleBuilder_h_
#define Kassiopeia_KSCommandMemberSimpleBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandMember.h"
#include "KToolbox.h"

#include <type_traits>

using namespace Kassiopeia;

namespace katrin
{
class KSCommandMemberSimpleData
{
  public:
    KSCommandMemberSimpleData() : fName(""), fParentName(""), fChildName(""), fFieldName("") {}
    std::string fName;
    std::string fParentName;
    std::string fChildName;
    std::string fFieldName;
};

//marco definition,
// as all builder for adding or removing from the root toolboxes look the same, except for the names and std::strings
#define KSCOMMANDMEMBERSIMPLEBUILDERHEADER(xDEFAULTPARENTNAME, xFIELDNAME, xBUILDERNAME)                               \
                                                                                                                       \
    using namespace std;                                                                                               \
                                                                                                                       \
    class KSCommandMember##xBUILDERNAME##Data : public KSCommandMemberSimpleData                                       \
    {};                                                                                                                \
                                                                                                                       \
    typedef KComplexElement<KSCommandMember##xBUILDERNAME##Data> KSCommandMember##xBUILDERNAME##Builder;               \
                                                                                                                       \
    template<> inline bool KSCommandMember##xBUILDERNAME##Builder::Begin()                                             \
    {                                                                                                                  \
        fObject = new KSCommandMember##xBUILDERNAME##Data();                                                           \
        return true;                                                                                                   \
    }                                                                                                                  \
    template<> inline bool KSCommandMember##xBUILDERNAME##Builder::AddAttribute(KContainer* aContainer)                \
    {                                                                                                                  \
        if (aContainer->GetName() == "name") {                                                                         \
            std::string tName = aContainer->AsString();                                                                \
            fObject->fName = tName;                                                                                    \
            return true;                                                                                               \
        }                                                                                                              \
        if (aContainer->GetName() == "parent") {                                                                       \
            std::string tParent = aContainer->AsString();                                                              \
            fObject->fParentName = tParent;                                                                            \
            return true;                                                                                               \
        }                                                                                                              \
        if (aContainer->GetName() == "child") {                                                                        \
            std::string tComponent = aContainer->AsString();                                                           \
            fObject->fChildName = tComponent;                                                                          \
            return true;                                                                                               \
        }                                                                                                              \
        return false;                                                                                                  \
    }                                                                                                                  \
    template<> inline bool KSCommandMember##xBUILDERNAME##Builder::End()                                               \
    {                                                                                                                  \
        fObject->fFieldName = xFIELDNAME;                                                                              \
        if (fObject->fParentName.empty()) {                                                                            \
            fObject->fParentName = xDEFAULTPARENTNAME;                                                                 \
        }                                                                                                              \
        objctmsg_debug(#xBUILDERNAME << ": " << fObject->fParentName << " . " << fObject->fChildName << " : "          \
                                     << fObject->fFieldName << eom);                                                   \
        KSComponent* tParent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParentName);                         \
        if (tParent == nullptr) {                                                                                      \
            objctmsg(eError) << "command member <" << fObject->fName << "> could not find parent <"                    \
                             << fObject->fParentName << "> for builder <" << #xBUILDERNAME << ">" << eom;              \
        }                                                                                                              \
        KSComponent* tChild = KToolbox::GetInstance().Get<KSComponent>(fObject->fChildName);                           \
        if (tChild == nullptr) {                                                                                       \
            tChild = KToolbox::GetInstance().Get<KSComponent>(fObject->fChildName + "_"); /* try alias */              \
            if (tChild == nullptr) {                                                                                   \
                objctmsg(eError) << "command member <" << fObject->fName << "> could not find child <"                 \
                                 << fObject->fChildName << "> for builder <" << #xBUILDERNAME << ">" << eom;           \
            }                                                                                                          \
        }                                                                                                              \
        KSCommand* tCommand = tParent->Command(fObject->fFieldName, tChild);                                           \
        if (tCommand == nullptr) {                                                                                     \
            objctmsg(eError) << "command member <" << fObject->fName << "> could activate field <"                     \
                             << fObject->fFieldName << "> in parent <" << fObject->fParentName << "> for builder <"    \
                             << #xBUILDERNAME << ">" << eom;                                                           \
        }                                                                                                              \
        if (!fObject->fName.empty()) {                                                                                 \
            tCommand->SetName(fObject->fName);                                                                         \
        }                                                                                                              \
        delete fObject;                                                                                                \
        Set(tCommand);                                                                                                 \
        return true;                                                                                                   \
    }

//add/remove terminator
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_terminator", "add_terminator", AddTerminator);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_terminator", "remove_terminator", RemoveTerminator);

//add/remove magnetic field
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_magnetic_field", "add_magnetic_field", AddMagneticField);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_magnetic_field", "remove_magnetic_field", RemoveMagneticField);

//add/remove magnetic field
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_electric_field", "add_electric_field", AddElectricField);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_electric_field", "remove_electric_field", RemoveElectricField);

//add/remove space interaction
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_space_interaction", "add_space_interaction", AddSpaceInteraction);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_space_interaction", "remove_space_interaction", RemoveSpaceInteraction);

//set/clear density
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for set_density", "set_density", SetDensity);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for clear_density", "clear_density", ClearDensity);

//set/clear surface interaction
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_surface_interaction", "set_surface_interaction", SetSurfaceInteraction);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_surface_interaction", "clear_surface_interaction", ClearSurfaceInteraction);

//add/remove step modifier
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_stepmodifier", "add_modifier", AddStepModifier);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_stepmodifier", "remove_modifier", RemoveStepModifier);

//set/clear trajectory
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_trajectory", "set_trajectory", SetTrajectory);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("root_trajectory", "clear_trajectory", ClearTrajectory);

//add/remove control
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_control", "add_control", AddControl);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_control", "remove_control", RemoveControl);

//add/remove term
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_term", "add_term", AddTerm);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_Term", "remove_term", RemoveTerm);

//add/remove step output
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_step_output", "add_step_output", AddStepOutput);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_step_output", "remove_step_output", RemoveStepOutput);

//add/remove track output
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_track_output", "add_track_output", AddTrackOutput);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_track_output", "remove_track_output",
                                   RemoveTrackOutput);

//add/remove step write condition
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_step_write_condition", "add_step_write_condition",
                                   AddStepWriteCondition);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_step_write_condition", "remove_step_write_condition",
                                   RemoveStepWriteCondition);

//add/remove track write condition
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for add_track_write_condition", "add_track_write_condition",
                                   AddTrackWriteCondition);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for remove_track_write_condition", "remove_track_write_condition",
                                   RemoveTrackWriteCondition);

//set/clear vtk step point
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for set_step_point", "set_step_point", SetStepPoint);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for clear_step_point", "clear_step_point", ClearStepPoint);

//set/clear vtk step data
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for set_step_data", "set_step_data", SetStepData);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for clear_step_data", "clear_step_data", ClearStepData);

//set/clear vtk track point
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for set_track_point", "set_track_point", SetTrackPoint);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for clear_track_point", "clear_track_point", ClearTrackPoint);

//set/clear vtk track data
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for set_track_data", "set_track_data", SetTrackData);
KSCOMMANDMEMBERSIMPLEBUILDERHEADER("no default parent for clear_track_data", "clear_track_data", ClearTrackData);


#undef KSCOMMANDMEMBERSIMPLEBUILDERHEADER

}  // namespace katrin

#endif
