#ifndef KOPENCLINTERFACE_DEF
#define KOPENCLINTERFACE_DEF

#ifdef KEMFIELD_USE_CL_VECTOR
#define CL_HPP_NO_STD_VECTOR  // Use cl::vector instead of STL version
#define CL_VECTOR_TYPE cl::vector
#else
#define CL_VECTOR_TYPE std::vector
#endif

#define CL_HPP_ENABLE_EXCEPTIONS

//shut up the annoying unused variable warnings in cl.hpp for clang/gcc
//by using a system-header wrapper for the opencl headers
#include "KOpenCLHeaderWrapper.hh"

// #ifdef GCC_VERSION
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wall -Wcomment"
// #endif
//
// #ifdef __clang__
// #pragma clang diagnostic push
// #pragma clang system_header
// #endif
//
// #if defined __APPLE__
// #include <OpenCL/cl.hpp>
// #else
// #pragma GCC system_header
// #include <CL/cl.hpp>
// #endif
//
// //turn warnings back on
// #ifdef __clang__
// #pragma clang diagnostic pop
// #endif
//
// #ifdef GCC_VERSION
// #pragma GCC diagnostic pop
// #endif

#undef CL_TYPE
#undef CL_TYPE2
#undef CL_TYPE4
#undef CL_TYPE8
#undef CL_TYPE16

#ifdef KEMFIELD_USE_DOUBLE_PRECISION
#define CL_TYPE   cl_double
#define CL_TYPE2  cl_double2
#define CL_TYPE4  cl_double4
#define CL_TYPE8  cl_double8
#define CL_TYPE16 cl_double16
#else
#define CL_TYPE   cl_float
#define CL_TYPE2  cl_float2
#define CL_TYPE4  cl_float4
#define CL_TYPE8  cl_float8
#define CL_TYPE16 cl_float16
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif

#include <vector>

//this is necessary on some intel devices
#define ENFORCE_CL_FINISH

//the following are optional defines for debugging

//adds verbose output of kernel build logs
//#define DEBUG_OPENCL_COMPILER_OUTPUT

//add try-catch for opencl exceptions
//#define USE_CL_ERROR_TRY_CATCH

#ifdef USE_CL_ERROR_TRY_CATCH
#define CL_ERROR_TRY try
#else
#define CL_ERROR_TRY
#endif

#ifdef USE_CL_ERROR_TRY_CATCH
#define CL_ERROR_CATCH                                                                                                 \
    catch (cl::Error & error)                                                                                          \
    {                                                                                                                  \
        std::cout << "OpenCL Exception caught: " << std::endl;                                                         \
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;                                                         \
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;                                           \
        std::exit(1);                                                                                                  \
    }
#else
#define CL_ERROR_CATCH
#endif


namespace KEMField
{

class KOpenCLData;

class KOpenCLInterface
{
  public:
    static KOpenCLInterface* GetInstance();

    void Initialize();

    cl::Context GetContext() const
    {
        return *fContext;
    }
    CL_VECTOR_TYPE<cl::Device> GetDevices() const
    {
        return fDevices;
    }
    cl::Device GetDevice() const
    {
        return fDevices.at(fCLDeviceID);
    }
    cl::CommandQueue& GetQueue(int i = -1) const;

    unsigned int GetNumberOfDevices() const
    {
        if (! fContext)
            return 0;

        CL_VECTOR_TYPE<cl::Device> availableDevices = fContext->getInfo<CL_CONTEXT_DEVICES>();
        return availableDevices.size();
    };

    void SetGPU(unsigned int i);

    void SetKernelPath(std::string s)
    {
        fKernelPath = s;
    }
    std::string GetKernelPath() const
    {
        return fKernelPath;
    }

    void SetActiveData(KOpenCLData* data);
    KOpenCLData* GetActiveData() const;

  protected:
    KOpenCLInterface();
    virtual ~KOpenCLInterface();

    static KOpenCLInterface* fOpenCLInterface;

    std::string fKernelPath;

    CL_VECTOR_TYPE<cl::Platform> fPlatforms;
    CL_VECTOR_TYPE<cl::Device> fDevices;
    unsigned int fCLDeviceID;
    cl::Context* fContext;
    mutable std::vector<cl::CommandQueue*> fQueues;
    mutable KOpenCLData* fActiveData;
};

}  // namespace KEMField

#endif /* KOPENCLINTERFACE_DEF */
