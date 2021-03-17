#include "KSCommandMemberSimpleBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
//macro for the destructor and the attributes
#define KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(xBUILDERNAME)                                                               \
                                                                                                                       \
    template<> KSCommandMember##xBUILDERNAME##Builder::~KComplexElement() {}                                           \
                                                                                                                       \
    STATICINT sKSCommand##xBUILDERNAME##Structure =                                                                    \
        KSCommandMember##xBUILDERNAME##Builder::Attribute<std::string>("name") +                                       \
        KSCommandMember##xBUILDERNAME##Builder::Attribute<std::string>("parent") +                                     \
        KSCommandMember##xBUILDERNAME##Builder::Attribute<std::string>("child");

//add/remove terminator
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddTerminator);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveTerminator);

//add/remove magnetic field
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddMagneticField);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveMagneticField);

//add/remove electric field
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddElectricField);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveElectricField);

//add/remove space interaction
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddSpaceInteraction);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveSpaceInteraction);

//set/clear density
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetDensity);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearDensity);

//set/clear surface interaction
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetSurfaceInteraction);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearSurfaceInteraction);

//add/remove step modifier
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddStepModifier);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveStepModifier);

//set/clear trajectory
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetTrajectory);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearTrajectory);

//add/remove control
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddControl);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveControl);

//add/remove term
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddTerm);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveTerm);

//add/remove step output
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddStepOutput);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveStepOutput);

//add/remove track output
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddTrackOutput);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveTrackOutput);

//add/remove step write condition
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddStepWriteCondition);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveStepWriteCondition);

//add/remove track write condition
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(AddTrackWriteCondition);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(RemoveTrackWriteCondition);

//set/clear vtk step point
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetStepPoint);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearStepPoint);

//set/clear vtk step data
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetStepData);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearStepData);

//set/clear vtk track point
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetTrackPoint);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearTrackPoint);

//set/clear vtk track data
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(SetTrackData);
KSCOMMANDMEMBERSIMPLEBUILDERSOURCE(ClearTrackData);

#undef KSCOMMANDMEMBERSIMPLEBUILDERSOURCE

}  // namespace katrin
