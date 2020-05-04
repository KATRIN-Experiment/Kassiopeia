#include "KOpenCLAction.hh"

namespace KEMField
{
KOpenCLAction::KOpenCLAction(KOpenCLData& data, bool enabled) : fData(data), fEnabled(enabled)
{
    fData.RegisterAction(this);
}

void KOpenCLAction::Initialize()
{
    if (!(fData.IsConstructed())) {
        // If the container has not yet been built, then kernel and buffer
        // construction will occur during its construction.
        fData.ConstructOpenCLObjects();
    }
    else {
        // If the container has already been built, we can still try to construct
        // this action and associate it with the data's buffers.  There may
        // be issues with workgroup sizing, though.
        ConstructOpenCLKernels();
        AssignBuffers();
    }
}
}  // namespace KEMField
