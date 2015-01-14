#ifndef KOPENCLINTERFACE_DEF
#define KOPENCLINTERFACE_DEF

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#if defined __APPLE__

#ifdef __clang__ //shut up the annoying unused variable warnings in cl.hpp
#pragma clang diagnostic push
#pragma clang system_header
#endif

#include <OpenCL/cl.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else

#ifdef __clang__  //shut up the annoying unused variable  warnings in cl.hpp
#pragma clang diagnostic push
#pragma clang system_header
#endif

#include <CL/cl.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
#if defined KEMFIELD_USE_DOUBLE_PRECISION
#define CL_TYPE cl_double
#define CL_TYPE2 cl_double2
#define CL_TYPE4 cl_double4
#define CL_TYPE8 cl_double8
#define CL_TYPE16 cl_double16
#else
#define CL_TYPE cl_float
#define CL_TYPE2 cl_float2
#define CL_TYPE4 cl_float4
#define CL_TYPE8 cl_float8
#define CL_TYPE16 cl_float16
#endif

#include <vector>

namespace KEMField{

  class KOpenCLData;

  class KOpenCLInterface
  {
  public:
    static KOpenCLInterface* GetInstance();

    cl::Context            GetContext() const { return *fContext; }
    cl::vector<cl::Device> GetDevices() const { return fDevices; }
    cl::Device             GetDevice()  const { return fDevices[fCLDeviceID]; }
    cl::CommandQueue&      GetQueue(int i=-1) const;

    void SetGPU(unsigned int i);

    void SetKernelPath(std::string s) { fKernelPath = s; }
    std::string GetKernelPath() const { return fKernelPath; }

    void SetActiveData(KOpenCLData* data);
    KOpenCLData* GetActiveData() const;

  protected:
    KOpenCLInterface();
    virtual ~KOpenCLInterface();

    void InitializeOpenCL();

    static KOpenCLInterface* fOpenCLInterface;

    std::string fKernelPath;

    cl::vector<cl::Platform> fPlatforms;
    cl::vector<cl::Device>   fDevices;
    unsigned int                   fCLDeviceID;
    cl::Context              *fContext;
    mutable std::vector<cl::CommandQueue*>  fQueues;
    mutable KOpenCLData *fActiveData;
  };

}

#endif /* KOPENCLINTERFACE_DEF */
