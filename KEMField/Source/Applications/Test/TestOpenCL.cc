#include "mpi.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

//#define CL_HPP_NO_STD_VECTOR  // Use cl::vector instead of STL version
#define CL_HPP_ENABLE_EXCEPTIONS

#if defined __APPLE__
#include <OpenCL/cl2.hpp>
#else
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/cl2.hpp>
#endif

#ifndef KEMFIELD_DEFAULT_GPU_ID
#define KEMFIELD_DEFAULT_GPU_ID 0
#endif /* KEMFIELD_DEFAULT_GPU_ID */

#ifndef DEFAULT_KERNEL_DIR
#define DEFAULT_KERNEL_DIR "./"
#endif /* !DEFAULT_KERNEL_DIR */

namespace KEMField
{

class KOpenCLInterface
{
  public:
    static KOpenCLInterface* GetInstance();

    cl::Context GetContext() const
    {
        return *fContext;
    }
    cl::vector<cl::Device> GetDevices() const
    {
        return fDevices;
    }
    size_t GetNumberOfDevices() const
    {
        return fDevices.size();
    }
    cl::Device GetDevice() const
    {
        return fDevices[fCLDeviceID];
    }
    cl::CommandQueue& GetQueue(int i = -1) const;

    void SetGPU(unsigned int i);

    void SetKernelPath(std::string s)
    {
        fKernelPath = s;
    }
    std::string GetKernelPath() const
    {
        return fKernelPath;
    }

  protected:
    KOpenCLInterface();
    virtual ~KOpenCLInterface();

    void InitializeOpenCL();

    static KOpenCLInterface* fOpenCLInterface;

    std::string fKernelPath;

    cl::vector<cl::Platform> fPlatforms;
    cl::vector<cl::Device> fDevices;
    unsigned int fCLDeviceID;
    cl::Context* fContext;
    mutable std::vector<cl::CommandQueue*> fQueues;
};

KOpenCLInterface* KOpenCLInterface::fOpenCLInterface = 0;

KOpenCLInterface::KOpenCLInterface()
{
    InitializeOpenCL();
}

KOpenCLInterface::~KOpenCLInterface()
{
    for (std::vector<cl::CommandQueue*>::iterator it = fQueues.begin(); it != fQueues.end(); ++it)
        if (*it)
            delete *it;
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
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(fPlatforms[0])(), 0};
    fContext = new cl::Context(CL_DEVICE_TYPE_GPU, cps);

    cl::vector<cl::Device> availableDevices = fContext->getInfo<CL_CONTEXT_DEVICES>();

    fCLDeviceID = KEMFIELD_DEFAULT_GPU_ID;
    fDevices.clear();
    fDevices = availableDevices;
    fQueues.resize(fDevices.size(), NULL);

    fKernelPath = DEFAULT_KERNEL_DIR;
}

/**
   * Selects a device for use in OpenCL calculations.
   */
void KOpenCLInterface::SetGPU(unsigned int i)
{
    if (i >= fDevices.size()) {
        std::stringstream s;
        s << "Cannot set GPU device to ID # " << i << ", since there are only " << fDevices.size()
          << " devices available.";
        std::cout << s.str() << std::endl;
        return;
    }
    cl::STRING_CLASS extensions;
    fDevices[i].getInfo(CL_DEVICE_EXTENSIONS, &extensions);
    if ((extensions.find("cl_khr_fp64") == std::string::npos) &&
        (extensions.find("cl_amd_fp64") == std::string::npos)) {
        std::stringstream s;
        s << "Cannot set GPU device to ID # " << i
          << ", since it does not support double precision (and this program was built with double precision enabled).";
        std::cout << s.str() << std::endl;
        return;
    }

    cl::STRING_CLASS name;
    fDevices[i].getInfo(CL_DEVICE_NAME, &name);
    std::stringstream s;
    s << "Setting GPU device to ID # " << i << " (" << name << "), " << fDevices.size() << " available devices";
    s << " (double precision enabled).";
    std::cout << s.str() << std::endl;

    fCLDeviceID = i;
}

cl::CommandQueue& KOpenCLInterface::GetQueue(int i) const
{
    int deviceID = (i == -1 ? fCLDeviceID : i);

    if (!fQueues.at(deviceID))
        fQueues[deviceID] = new cl::CommandQueue(GetContext(), GetDevice());
    return *fQueues[deviceID];
}


class KMPIInterface
{
  public:
    static KMPIInterface* GetInstance();

    void Initialize(int* argc, char*** argv);
    void Finalize();

    int GetProcess() const
    {
        return fProcess;
    }
    int GetNProcesses() const
    {
        return fNProcesses;
    }

    void BeginSequentialProcess();
    void EndSequentialProcess();

    void GlobalBarrier() const
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

  protected:
    KMPIInterface();
    virtual ~KMPIInterface();

    static KMPIInterface* fMPIInterface;

    int fProcess;
    int fNProcesses;
    MPI_Status fStatus;
};

KMPIInterface* KMPIInterface::fMPIInterface = 0;

KMPIInterface::KMPIInterface()
{
    fProcess = -1;
    fNProcesses = -1;
}

KMPIInterface::~KMPIInterface() {}

void KMPIInterface::Initialize(int* argc, char*** argv)
{
    /* Let the system do what it needs to start up MPI */
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized)
        MPI_Init(argc, argv);

    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &fProcess);

    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &fNProcesses);
}

void KMPIInterface::Finalize()
{
    /* Shut down MPI */
    int finalized = 0;
    MPI_Finalized(&finalized);

    if (!finalized)
        MPI_Finalize();
}

/**
   * Interface to accessing KMPIInterface.
   */
KMPIInterface* KMPIInterface::GetInstance()
{
    if (fMPIInterface == 0)
        fMPIInterface = new KMPIInterface();
    return fMPIInterface;
}

/**
   * Ensures that a process written between BeginSequentialProcess() and
   * EndSequentialProcess() is done one processor at a time.
   */
void KMPIInterface::BeginSequentialProcess()
{
    int flag = 1;

    if (fProcess > 0)
        MPI_Recv(&flag, 1, MPI_INT, fProcess - 1, 50, MPI_COMM_WORLD, &fStatus);
}

/**
   * @see BeginSequentialProcess()
   */
void KMPIInterface::EndSequentialProcess()
{
    int flag;

    if (fProcess < (fNProcesses - 1))
        MPI_Send(&flag, 1, MPI_INT, fProcess + 1, 50, MPI_COMM_WORLD);
}
}  // namespace KEMField

using namespace KEMField;

#define DATA_SIZE (10240)

const char* KernelSource = "\n"
                           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                           \n"
                           "                                                                        \n"
                           "double ArcSinh(double x)                                                \n"
                           "{                                                                       \n"
                           "  return log(x + sqrt(1. + x*x));                                       \n"
                           "}                                                                       \n"
                           "                                                                        \n"
                           "__kernel void vector_ArcSinh(                                           \n"
                           "   __global double* input,                                              \n"
                           "   __global double* output,                                             \n"
                           "   __global int* count)                                                 \n"
                           "{                                                                       \n"
                           "   int i = get_global_id(0);                                            \n"
                           "   if (i<count[0])                                                      \n"
                           "     output[i] = ArcSinh(input[i]);                                     \n"
                           "}                                                                       \n"
                           "\n";


int main(int argc, char** argv)
{
    KMPIInterface::GetInstance()->Initialize(&argc, &argv);

    KMPIInterface::GetInstance()->BeginSequentialProcess();

    std::cout << "---" << std::endl;
    std::cout << "Available GPUs: KOpenCLInterface::GetInstance()->GetDevices().size()         = "
              << KOpenCLInterface::GetInstance()->GetDevices().size() << std::endl;
    std::cout << "Total number of MPI processes: KMPIInterface::GetInstance()->GetNProcesses() = "
              << KMPIInterface::GetInstance()->GetNProcesses() << std::endl;
    std::cout << "Process no. " << KMPIInterface::GetInstance()->GetProcess() + 1 << " is starting." << std::endl;
    //std::cout << "Process " << KMPIInterface::GetInstance()->GetProcess()<< " of " << KMPIInterface::GetInstance()->GetNProcesses() << " is starting." << std::endl;
    //std::cout << "Using gpu # " << KMPIInterface::GetInstance()->GetProcess()%2 << std::endl;
    std::cout << "---" << std::endl;

    //assign devices according to the number available and local process rank
    unsigned int proc_id = KMPIInterface::GetInstance()->GetProcess();
    int n_dev = KOpenCLInterface::GetInstance()->GetNumberOfDevices();
    int dev_id = proc_id % n_dev;  //fallback to global process rank if local is unavailable
//    int local_rank = KMPIInterface::GetInstance()->GetLocalRank();
//    if (local_rank != -1) {
//        if (KMPIInterface::GetInstance()->SplitMode()) {
//            dev_id = (local_rank / 2) % n_dev;
//        }
//        else {
//            dev_id = (local_rank) % n_dev;
//        }
//    }
    std::cout << "Setting GPU device to " << dev_id << "." << std::endl;
    KOpenCLInterface::GetInstance()->SetGPU(dev_id);

    //KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());

    KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());

    int* count = new int[1];
    count[0] = DATA_SIZE;

    double arcSinhInputs[DATA_SIZE];
    double arcSinhOutputs[DATA_SIZE];

    for (unsigned int i = 0; i < DATA_SIZE; i++)
        arcSinhInputs[i] = sinh(((double) i) / DATA_SIZE);

    // Source file
    std::string sourceCode = KernelSource;

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(KOpenCLInterface::GetInstance()->GetContext(), source);

    // Build program for these specific devices
    try {
        cl::vector<cl::Device> devices;
        devices.push_back(KOpenCLInterface::GetInstance()->GetDevice());
        program.build(devices, "");
    }
    catch (cl::Error& error) {
        std::cerr << "There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"
                  << std::endl;
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::cerr << "Build Status: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())
                  << std::endl;
        std::cerr << "Build Options:\t"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())
                  << std::endl;
        std::cerr << "Build Log:\t "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());
        return 0;
    }

    // Make kernel
    cl::Kernel kernel(program, "vector_ArcSinh");

    // Create memory buffers
    cl::Buffer* bufferIn;
    cl::Buffer* bufferOut;
    bufferIn =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, DATA_SIZE * sizeof(double));
    bufferOut =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, DATA_SIZE * sizeof(double));

    cl::Buffer bufferCount = cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sizeof(int));

    // Copy in buffer to the memory buffer
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*bufferIn,
                                                                   CL_TRUE,
                                                                   0,
                                                                   DATA_SIZE * sizeof(double),
                                                                   arcSinhInputs);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(bufferCount, CL_TRUE, 0, sizeof(int), count);

    // Set arguments to kernel
    kernel.setArg(0, *bufferIn);
    kernel.setArg(1, *bufferOut);
    kernel.setArg(2, bufferCount);

    // Run the kernel on specific ND range
    cl::NDRange global(DATA_SIZE);
    cl::NDRange local(1);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    // Read out buffer into a local list
    cl::Event event;
    KOpenCLInterface::GetInstance()
        ->GetQueue()
        .enqueueReadBuffer(*bufferOut, CL_TRUE, 0, DATA_SIZE * sizeof(double), arcSinhOutputs, NULL, &event);

    event.wait();

    for (unsigned int i = 0; i < DATA_SIZE; i++) {
        std::cout << "Process no. " << KMPIInterface::GetInstance()->GetProcess() + 1 << " : arcsinh(sinh("
                  << (double) i / DATA_SIZE << ")) = " << arcSinhOutputs[i] << std::endl;
    }

    std::cout << "Process no. " << KMPIInterface::GetInstance()->GetProcess() + 1 << " is ending." << std::endl;

    KMPIInterface::GetInstance()->EndSequentialProcess();

    KMPIInterface::GetInstance()->Finalize();

    return 0;
}
