#include "KOpenCLInterface.hh"

#include "KOpenCLData.hh"

#include "KEMCout.hh"

#include <stdlib.h>
#include <sstream>


#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
#endif

#define KEMFIELD_DEFAULT_GPU_ID 0

#ifndef DEFAULT_KERNEL_DIR
#define DEFAULT_KERNEL_DIR "."
#endif /* !DEFAULT_KERNEL_DIR */

namespace KEMField
{
  KOpenCLInterface* KOpenCLInterface::fOpenCLInterface = 0;

  KOpenCLInterface::KOpenCLInterface() : fActiveData(NULL)
  {
    InitializeOpenCL();
  }

  KOpenCLInterface::~KOpenCLInterface()
  {
    for (std::vector<cl::CommandQueue*>::iterator it=fQueues.begin();it!=fQueues.end();++it)
      if (*it) delete *it;
  }

  /**
   * Interface to accessing KOpenCLInterface.
   */
  KOpenCLInterface* KOpenCLInterface::GetInstance()
  {
    if (fOpenCLInterface == 0)
      fOpenCLInterface = new KOpenCLInterface();
    return fOpenCLInterface;
  }

  /**
   * Queries the host for available OpenCL platforms, and constructs a Context.
   */
  void KOpenCLInterface::InitializeOpenCL()
  {
    // Disable CUDA caching, since it doesn't check if included .cl files have
    // changed
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    // Get available platforms
    cl::Platform::get(&fPlatforms);

    // Select the default platform and create a context using this platform and
    // the GPU
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
				    (cl_context_properties)(fPlatforms[KEMFIELD_OPENCL_PLATFORM])(),
				    0};

    unsigned int deviceType = KEMFIELD_OPENCL_DEVICE_TYPE;


    if(deviceType == 0) //we have a GPU
    {
        fContext = new cl::Context( CL_DEVICE_TYPE_GPU, cps);
    }

    if(deviceType == 1) //we have a CPU
    {
        fContext = new cl::Context( CL_DEVICE_TYPE_CPU, cps);
    }

    if(deviceType == 2) //we have an accelerator device
    {
        fContext = new cl::Context( CL_DEVICE_TYPE_ACCELERATOR, cps);
    }

    CL_VECTOR_TYPE<cl::Device> availableDevices = fContext->getInfo<CL_CONTEXT_DEVICES>();

    fCLDeviceID = KEMFIELD_DEFAULT_GPU_ID;
    fDevices.clear();
    fDevices = availableDevices;
    fQueues.resize(fDevices.size(),NULL);

    fKernelPath = DEFAULT_KERNEL_DIR;
  }

  /**
   * Selects a device for use in OpenCL calculations.
   */
  void KOpenCLInterface::SetGPU(unsigned int i)
  {
    if (i>=fDevices.size())
    {
      KEMField::cout << "Cannot set GPU device to ID # "<<i<<", since there are only "<<fDevices.size()<<" devices available." << KEMField::endl;
      return;
    }
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    cl::STRING_CLASS extensions;
    fDevices[i].getInfo(CL_DEVICE_EXTENSIONS, &extensions);
    if ((extensions.find("cl_khr_fp64") == std::string::npos) &&
	(extensions.find("cl_amd_fp64") == std::string::npos))
    {
      KEMField::cout << "Cannot set GPU device to ID # "<<i<<", since it does not support double precision (and this program was built with double precision enabled)." << KEMField::endl;
      return;
    }
#endif /* KEMFIELD_USE_DOUBLE_PRECISION */


#ifdef KEMFIELD_USE_MPI

    cl::STRING_CLASS name;
    int process_id = KMPIInterface::GetInstance()->GetProcess();
    fDevices[i].getInfo(CL_DEVICE_NAME, &name);
    std::stringstream msg;
    msg << "Process #"<<process_id<<", Setting GPU device to ID # "<<i<<" ("<<name<<") of "<<fDevices.size()<<" available devices on host: "<< KMPIInterface::GetInstance()->GetHostName();
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    msg << " (double precision enabled)." << std::endl;
#else
    msg << "." << std::endl;
#endif

    KMPIInterface::GetInstance()->PrintMessage(msg.str());


#else

    cl::STRING_CLASS name;
    fDevices[i].getInfo(CL_DEVICE_NAME, &name);
    KEMField::cout << "Setting GPU device to ID # "<<i<<" ("<<name<<") of "<<fDevices.size()<<" available devices";
#ifdef KEMFIELD_USE_DOUBLE_PRECISION
    KEMField::cout << " (double precision enabled)." << KEMField::endl;
#else
    KEMField::cout << "." << KEMField::endl;
#endif

#endif

    fCLDeviceID = i;
  }

  void KOpenCLInterface::SetActiveData(KOpenCLData* data)
  {
    fActiveData = data;
  }

  KOpenCLData* KOpenCLInterface::GetActiveData() const
  {
    return fActiveData;
  }

  cl::CommandQueue& KOpenCLInterface::GetQueue(int i) const
  {
    int deviceID = (i == -1 ? fCLDeviceID : i);

    if (!fQueues.at(deviceID))
      fQueues[deviceID] = new cl::CommandQueue(GetContext(),GetDevice());
    return *fQueues[deviceID];
  }
}
