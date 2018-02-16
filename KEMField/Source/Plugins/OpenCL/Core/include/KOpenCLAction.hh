#ifndef KOPENCLACTION_DEF
#define KOPENCLACTION_DEF

#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"
#include "KOpenCLData.hh"
#include "KOpenCLInterface.hh"

namespace KEMField
{

/**
* @class KOpenCLAction
*
* @brief A class that acts on KOpenCLData.
*
* KOpenCLAction is a parent class for objects that contain kernels designed to
* manipulate data in a KOpenCLData object.  The purpose is to control the order
* in which OpenCL-related objects are initialized; the lengths of the buffers in
* a KOpenCLData are dependent on the memory available on the compute device,
* which must be determined after the kernel is compiled and before the buffers
* are assigned to the kernel.
*
* @author T.J. Corona
*/

  class KOpenCLAction
  {
  public:
    KOpenCLAction(KOpenCLData& data,bool enabled = true);
    virtual ~KOpenCLAction() {}

    virtual void Initialize();

    virtual void ConstructOpenCLKernels() const = 0;
    virtual void AssignBuffers() const = 0;

    virtual std::string GetOpenCLFlags() const { return std::string(""); }

    KOpenCLData& GetData() const { return fData; }

    void Enabled(bool b) { fEnabled = b; }
    bool Enabled() const { return fEnabled; }

  protected:
    KOpenCLData& fData;

    bool fEnabled;
  };
}

#endif /* KOPENCLACTION_DEF */
