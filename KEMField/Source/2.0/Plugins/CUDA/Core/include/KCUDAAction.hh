#ifndef KCUDAACTION_DEF
#define KCUDAACTION_DEF

#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"
#include "KCUDAData.hh"
#include "KCUDAInterface.hh"

namespace KEMField
{

/**
* @class KCUDAAction
*
* @brief A class that acts on KCUDAData.
*
* KCUDAAction is a parent class for objects that contain kernels designed to
* manipulate data in a KCUDAData object.  The purpose is to control the order
* in which CUDA-related objects are initialized; the lengths of the buffers in
* a KCUDAData are dependent on the memory available on the compute device,
* which must be determined after the kernel is compiled and before the buffers
* are assigned to the kernel. Class adapted from corresponding OpenCL version from T.J. Corona.
*
* @author Daniel Hilk
*/

  class KCUDAAction
  {
  public:
    KCUDAAction(KCUDAData& data,bool enabled = true);
    virtual ~KCUDAAction() {}

    virtual void Initialize();

    virtual void ConstructCUDAKernels() const = 0;
    virtual void AssignDeviceMemory() const = 0;

    KCUDAData& GetData() const { return fData; }

    void Enabled(bool b) { fEnabled = b; }
    bool Enabled() const { return fEnabled; }

  protected:
    KCUDAData& fData;

    bool fEnabled;
  };
}

#endif /* KCUDAACTION_DEF */
