#include "KCUDAAction.hh"

namespace KEMField
{
  KCUDAAction::KCUDAAction(KCUDAData& data, bool enabled) :
    fData(data),
    fEnabled(enabled)
  {
    fData.RegisterAction(this);
  }

  void KCUDAAction::Initialize()
  {
    if (!(fData.IsConstructed()))
    {
      // If the container has not yet been built, then kernel and buffer
      // construction will occur during its construction.
      fData.ConstructCUDAObjects();
    }
    else
    {
      // If the container has already been built, we can still try to construct
      // this action and associate it with the data's buffers.  There may
      // be issues with workgroup sizing, though.
      ConstructCUDAKernels();
      AssignDeviceMemory();
    }
  }
}
