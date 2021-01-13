#include "KOpenCLData.hh"

#include "KEMCout.hh"
#include "KOpenCLAction.hh"

#include <limits>

namespace KEMField
{
KOpenCLData::KOpenCLData() : fNLocal(std::numeric_limits<unsigned int>::max()), fIsConstructed(false) {}

void KOpenCLData::ConstructOpenCLObjects()
{
    // Do nothing if this method has already been called
    if (fIsConstructed)
        return;

    // First, construct the kernels that will use the surface buffers (this sets
    // the number of dummy elements in each buffer)
    for (auto& associatedAction : fAssociatedActions) {
        if (associatedAction->Enabled())
            associatedAction->ConstructOpenCLKernels();
    }

    if (fNLocal == std::numeric_limits<unsigned int>::max()) {
        std::cout << "You must first set the number of local streams in a warp before setting the\nbuffers."
                  << std::endl;
        return;
    }

    // Then, build the actual objects
    BuildOpenCLObjects();

    // Finally, we assign the buffers to the actions
    for (auto& associatedAction : fAssociatedActions) {
        if (associatedAction->Enabled())
            associatedAction->AssignBuffers();
    }
}

void KOpenCLData::RegisterAction(const KOpenCLAction* action)
{
    fAssociatedActions.push_back(action);
}

void KOpenCLData::SetMinimumWorkgroupSizeForKernels(unsigned int nLocal)
{
    if (nLocal < fNLocal) {
        if (fIsConstructed)
            KEMField::cout << "Warning: Buffers have already been constructed with a workgroup size " << fNLocal << "."
                           << KEMField::endl;

        fNLocal = nLocal;
    }
}
}  // namespace KEMField
