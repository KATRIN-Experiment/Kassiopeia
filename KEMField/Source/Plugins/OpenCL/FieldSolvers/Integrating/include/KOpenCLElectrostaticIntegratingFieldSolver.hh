#ifndef KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF
#define KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KIntegratingFieldSolverTemplate.hh"
#include "KOpenCLAction.hh"
#include "KOpenCLSurfaceContainer.hh"

#include <fstream>
#include <limits.h>
#include <sstream>

#define MAX_SUBSET_SIZE 10000
#define MIN_SUBSET_SIZE 16

namespace KEMField
{
class ElectrostaticOpenCL;

template<class Integrator> class KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL> : public KOpenCLAction
{
  public:
    typedef typename Integrator::Basis Basis;

    KIntegratingFieldSolver(KOpenCLSurfaceContainer& container, Integrator& integrator);

    //use this constructor when sub-set solving is to be used
    KIntegratingFieldSolver(KOpenCLSurfaceContainer& container, Integrator& integrator, unsigned int max_subset_size,
                            unsigned int min_subset_size = 16);

    virtual ~KIntegratingFieldSolver();

    void ConstructOpenCLKernels() const;
    void AssignBuffers() const;

    double Potential(const KPosition& aPosition) const;
    KThreeVector ElectricField(const KPosition& aPosition) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KPosition& aPosition) const;

    ////////////////////////////////////////////////////////////////////////////
    //sub-set potential/field calls
    double Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    KThreeVector ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize,
                               const KPosition& aPosition) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize,
                                                              const KPosition& aPosition) const;

    //these methods allow us to dispatch a calculation to the GPU and retrieve the values later
    //this is useful so that we can do other work while waiting for the results
    void DispatchPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    void DispatchElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize,
                               const KPosition& aPosition) const;
    void DispatchElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize,
                                           const KPosition& aPosition) const;
    double RetrievePotential() const;
    KThreeVector RetrieveElectricField() const;
    std::pair<KThreeVector, double> RetrieveElectricFieldAndPotential() const;

    ////////////////////////////////////////////////////////////////////////////

    std::string GetOpenCLFlags() const
    {
        return fOpenCLFlags;
    }

  private:
    void GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids,
                                   unsigned int size) const;

    KOpenCLSurfaceContainer& fContainer;

    typedef typename Integrator::IntegratorSingleThread FallbackIntegrator;
    FallbackIntegrator fStandardIntegrator;
    KIntegratingFieldSolver<FallbackIntegrator> fStandardSolver;

    mutable std::string fOpenCLFlags;

    mutable cl::Kernel* fPotentialKernel;
    mutable cl::Kernel* fElectricFieldKernel;
    mutable cl::Kernel* fElectricFieldAndPotentialKernel;

    mutable cl::Buffer* fBufferP;
    mutable cl::Buffer* fBufferPotential;
    mutable cl::Buffer* fBufferElectricField;
    mutable cl::Buffer* fBufferElectricFieldAndPotential;

    mutable cl::NDRange* fGlobalRange;
    mutable cl::NDRange* fLocalRange;

    mutable CL_TYPE* fCLPotential;
    mutable CL_TYPE4* fCLElectricField;
    mutable CL_TYPE4* fCLElectricFieldAndPotential;

    mutable unsigned int fNGlobal;
    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    ////////////////////////////////////////////////////////////////////////////

    //sub-set kernel and buffers
    unsigned int fMaxSubsetSize;
    unsigned int fMinSubsetSize;
    mutable cl::Kernel* fSubsetPotentialKernel;
    mutable cl::Kernel* fSubsetElectricFieldKernel;
    mutable cl::Kernel* fSubsetElectricFieldAndPotentialKernel;
    mutable cl::Buffer* fBufferElementIdentities;
    mutable unsigned int* fSubsetIdentities;
    mutable unsigned int* fReorderedSubsetIdentities;

    //for dispatched and later collected subset potential/field
    mutable unsigned int fCachedNGlobal;
    mutable unsigned int fCachedNDummy;
    mutable unsigned int fCachedNWorkgroups;
    mutable double fCachedSubsetPotential;
    mutable KThreeVector fCachedSubsetField;
    mutable std::pair<KThreeVector, double> fCachedSubsetFieldAndPotential;

    mutable unsigned int fCachedSubsetSize;
    mutable const unsigned int* fCachedSurfaceIndexSet;
    mutable KPosition fCachedPosition;

    mutable bool fCallDevice;
};

template<class Integrator>
KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::KIntegratingFieldSolver(KOpenCLSurfaceContainer& container,
                                                                                  Integrator& integrator) :
    KOpenCLAction(container),
    fContainer(container),
    fStandardIntegrator(integrator.GetCPUIntegrator()),
    fStandardSolver(container.GetSurfaceContainer(), fStandardIntegrator),
    fPotentialKernel(NULL),
    fElectricFieldKernel(NULL),
    fElectricFieldAndPotentialKernel(NULL),
    fGlobalRange(NULL),
    fLocalRange(NULL),
    fCLPotential(NULL),
    fCLElectricField(NULL),
    fCLElectricFieldAndPotential(NULL),
    fMaxSubsetSize(MAX_SUBSET_SIZE),
    fMinSubsetSize(MIN_SUBSET_SIZE),
    fSubsetPotentialKernel(NULL),
    fSubsetElectricFieldKernel(NULL),
    fSubsetElectricFieldAndPotentialKernel(NULL),
    fBufferElementIdentities(NULL)
{
    std::stringstream options;
    options << container.GetOpenCLFlags() << " " << integrator.GetOpenCLFlags();
    fOpenCLFlags = options.str();
    fSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
    fReorderedSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
}

template<class Integrator>
KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::KIntegratingFieldSolver(KOpenCLSurfaceContainer& container,
                                                                                  Integrator& integrator,
                                                                                  unsigned int max_subset_size,
                                                                                  unsigned int min_subset_size) :
    KOpenCLAction(container),
    fContainer(container),
    fStandardIntegrator(integrator.GetCPUIntegrator()),
    fStandardSolver(container.GetSurfaceContainer(), fStandardIntegrator),
    fPotentialKernel(NULL),
    fElectricFieldKernel(NULL),
    fElectricFieldAndPotentialKernel(NULL),
    fGlobalRange(NULL),
    fLocalRange(NULL),
    fCLPotential(NULL),
    fCLElectricField(NULL),
    fCLElectricFieldAndPotential(NULL),
    fMaxSubsetSize(max_subset_size),
    fMinSubsetSize(min_subset_size),
    fSubsetPotentialKernel(NULL),
    fSubsetElectricFieldKernel(NULL),
    fSubsetElectricFieldAndPotentialKernel(NULL),
    fBufferElementIdentities(NULL)
{
    if (fMaxSubsetSize == 0) {
        fMaxSubsetSize = MAX_SUBSET_SIZE;
    };
    fSubsetIdentities = new unsigned int[fMaxSubsetSize];
    fReorderedSubsetIdentities = new unsigned int[fMaxSubsetSize];
    std::stringstream options;
    options << container.GetOpenCLFlags() << " " << integrator.GetOpenCLFlags();
    fOpenCLFlags = options.str();
}


template<class Integrator> KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::~KIntegratingFieldSolver()
{
    if (fPotentialKernel)
        delete fPotentialKernel;
    if (fElectricFieldKernel)
        delete fElectricFieldKernel;
    if (fElectricFieldAndPotentialKernel)
        delete fElectricFieldAndPotentialKernel;
    if (fSubsetPotentialKernel)
        delete fSubsetPotentialKernel;
    if (fSubsetElectricFieldKernel)
        delete fSubsetElectricFieldKernel;
    if (fSubsetElectricFieldAndPotentialKernel)
        delete fSubsetElectricFieldAndPotentialKernel;
    if (fGlobalRange)
        delete fGlobalRange;
    if (fLocalRange)
        delete fLocalRange;
    if (fCLPotential)
        delete fCLPotential;
    if (fCLElectricField)
        delete fCLElectricField;
    if (fCLElectricFieldAndPotential)
        delete fCLElectricFieldAndPotential;
    if (fBufferElementIdentities)
        delete fBufferElementIdentities;
    delete[] fSubsetIdentities;
    delete[] fReorderedSubsetIdentities;
}

template<class Integrator> void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::ConstructOpenCLKernels() const
{
    // Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath()
           << "/kEMField_ElectrostaticIntegratingFieldSolver_kernel.cl";

    // Read kernel source from file
    std::string sourceCode;
    std::ifstream sourceFile(clFile.str().c_str());

    sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    // Make program of the source code in the context
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source, 0);

    // Build program for these specific devices
    try {
        // use only target device!
        CL_VECTOR_TYPE<cl::Device> devices;
        devices.push_back(KOpenCLInterface::GetInstance()->GetDevice());
        program.build(devices, GetOpenCLFlags().c_str());
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"
                  << std::endl;
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        std::cout << "Build Status: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice()) << ""
                  << std::endl;
        std::cout << "Build Options:\t"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice()) << ""
                  << std::endl;
        std::cout << "Build Log:\t "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());
    }

#ifdef DEBUG_OPENCL_COMPILER_OUTPUT
    std::stringstream s;
    s << "Build Log for OpenCL file " << clFile.str() << " :\t ";
    std::stringstream build_log_stream;
    build_log_stream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())
                     << std::endl;
    std::string build_log;
    build_log = build_log_stream.str();
    if (build_log.size() != 0) {
        s << build_log;
        std::cout << s.str() << std::endl;
    }
#endif


    // Make kernels
    fPotentialKernel = new cl::Kernel(program, "Potential");
    fElectricFieldKernel = new cl::Kernel(program, "ElectricField");
    fElectricFieldAndPotentialKernel = new cl::Kernel(program, "ElectricFieldAndPotential");

    //sub-set kernels
    fSubsetPotentialKernel = new cl::Kernel(program, "SubsetPotential");
    fSubsetElectricFieldKernel = new cl::Kernel(program, "SubsetElectricField");
    fSubsetElectricFieldAndPotentialKernel = new cl::Kernel(program, "SubsetElectricFieldAndPotential");

    std::vector<cl::Kernel*> kernelArray;
    kernelArray.push_back(fPotentialKernel);
    kernelArray.push_back(fElectricFieldKernel);
    kernelArray.push_back(fElectricFieldAndPotentialKernel);
    kernelArray.push_back(fSubsetPotentialKernel);
    kernelArray.push_back(fSubsetElectricFieldKernel);
    kernelArray.push_back(fSubsetElectricFieldAndPotentialKernel);

    fNLocal = UINT_MAX;

    for (std::vector<cl::Kernel*>::iterator it = kernelArray.begin(); it != kernelArray.end(); ++it) {
        unsigned int workgroupSize =
            (*it)->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice()));
        if (workgroupSize < fNLocal)
            fNLocal = workgroupSize;
    }

    // make sure that the available local memory on the device is sufficient for
    // the workgroup size
    cl_ulong localMemory = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    if (fNLocal * sizeof(CL_TYPE4) > localMemory)
        fNLocal = localMemory / sizeof(CL_TYPE4);

    //ensure that fNLocal is a power of two (necessary for parallel reduction)
    unsigned int proper_size = 1;
    do {
        proper_size *= 2;
    } while (2 * proper_size <= fNLocal);
    fNLocal = proper_size;

    fContainer.SetMinimumWorkgroupSizeForKernels(fNLocal);
}

template<class Integrator> void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::AssignBuffers() const
{
    fNGlobal = fContainer.GetNBufferedElements();
    fNLocal = fContainer.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fContainer.GetNBufferedElements() / fNLocal;

    fGlobalRange = new cl::NDRange(fNGlobal);
    fLocalRange = new cl::NDRange(fNLocal);

    fBufferP = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, 3 * sizeof(CL_TYPE));
    fBufferPotential = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_WRITE_ONLY,
                                      fNWorkgroups * sizeof(CL_TYPE));
    fBufferElectricField = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                          CL_MEM_WRITE_ONLY,
                                          fNWorkgroups * sizeof(CL_TYPE4));
    fBufferElectricFieldAndPotential = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                      CL_MEM_WRITE_ONLY,
                                                      fNWorkgroups * sizeof(CL_TYPE4));

    fCLPotential = new CL_TYPE[fNWorkgroups];
    fCLElectricField = new CL_TYPE4[fNWorkgroups];
    fCLElectricFieldAndPotential = new CL_TYPE4[fNWorkgroups];

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferPotential,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNWorkgroups * sizeof(CL_TYPE),
                                                                   fCLPotential);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElectricField,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNWorkgroups * sizeof(CL_TYPE4),
                                                                   fCLElectricField);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElectricFieldAndPotential,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNWorkgroups * sizeof(CL_TYPE4),
                                                                   fCLElectricFieldAndPotential);

    // Set arguments to kernel
    fPotentialKernel->setArg(0, *fBufferP);
    fPotentialKernel->setArg(1, *fContainer.GetShapeInfo());
    fPotentialKernel->setArg(2, *fContainer.GetShapeData());
    fPotentialKernel->setArg(3, *fContainer.GetBasisData());
    fPotentialKernel->setArg(4, fNLocal * sizeof(CL_TYPE), NULL);
    fPotentialKernel->setArg(5, *fBufferPotential);

    fElectricFieldKernel->setArg(0, *fBufferP);
    fElectricFieldKernel->setArg(1, *fContainer.GetShapeInfo());
    fElectricFieldKernel->setArg(2, *fContainer.GetShapeData());
    fElectricFieldKernel->setArg(3, *fContainer.GetBasisData());
    fElectricFieldKernel->setArg(4, fNLocal * sizeof(CL_TYPE4), NULL);
    fElectricFieldKernel->setArg(5, *fBufferElectricField);

    fElectricFieldAndPotentialKernel->setArg(0, *fBufferP);
    fElectricFieldAndPotentialKernel->setArg(1, *fContainer.GetShapeInfo());
    fElectricFieldAndPotentialKernel->setArg(2, *fContainer.GetShapeData());
    fElectricFieldAndPotentialKernel->setArg(3, *fContainer.GetBasisData());
    fElectricFieldAndPotentialKernel->setArg(4, fNLocal * sizeof(CL_TYPE4), NULL);
    fElectricFieldAndPotentialKernel->setArg(5, *fBufferElectricFieldAndPotential);

    //create the element id buffer
    fBufferElementIdentities = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                              CL_MEM_READ_ONLY,
                                              fMaxSubsetSize * sizeof(unsigned int));

    fSubsetPotentialKernel->setArg(0, *fBufferP);
    fSubsetPotentialKernel->setArg(1, *fContainer.GetShapeInfo());
    fSubsetPotentialKernel->setArg(2, *fContainer.GetShapeData());
    fSubsetPotentialKernel->setArg(3, *fContainer.GetBasisData());
    fSubsetPotentialKernel->setArg(4, fNLocal * sizeof(CL_TYPE), NULL);
    fSubsetPotentialKernel->setArg(5, *fBufferPotential);
    fSubsetPotentialKernel->setArg(6, fMaxSubsetSize);
    fSubsetPotentialKernel->setArg(7, *fBufferElementIdentities);

    fSubsetElectricFieldKernel->setArg(0, *fBufferP);
    fSubsetElectricFieldKernel->setArg(1, *fContainer.GetShapeInfo());
    fSubsetElectricFieldKernel->setArg(2, *fContainer.GetShapeData());
    fSubsetElectricFieldKernel->setArg(3, *fContainer.GetBasisData());
    fSubsetElectricFieldKernel->setArg(4, fNLocal * sizeof(CL_TYPE4), NULL);
    fSubsetElectricFieldKernel->setArg(5, *fBufferElectricField);
    fSubsetElectricFieldKernel->setArg(6, fMaxSubsetSize);
    fSubsetElectricFieldKernel->setArg(7, *fBufferElementIdentities);

    fSubsetElectricFieldAndPotentialKernel->setArg(0, *fBufferP);
    fSubsetElectricFieldAndPotentialKernel->setArg(1, *fContainer.GetShapeInfo());
    fSubsetElectricFieldAndPotentialKernel->setArg(2, *fContainer.GetShapeData());
    fSubsetElectricFieldAndPotentialKernel->setArg(3, *fContainer.GetBasisData());
    fSubsetElectricFieldAndPotentialKernel->setArg(4, fNLocal * sizeof(CL_TYPE4), NULL);
    fSubsetElectricFieldAndPotentialKernel->setArg(5, *fBufferElectricFieldAndPotential);
    fSubsetElectricFieldAndPotentialKernel->setArg(6, fMaxSubsetSize);
    fSubsetElectricFieldAndPotentialKernel->setArg(7, *fBufferElementIdentities);
}

template<class Integrator>
double KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::Potential(const KPosition& aPosition) const
{
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fPotentialKernel,
                                                                     cl::NullRange,
                                                                     *fGlobalRange,
                                                                     *fLocalRange);

    CL_TYPE potential = 0.;

    cl::Event event;
    KOpenCLInterface::GetInstance()
        ->GetQueue()
        .enqueueReadBuffer(*fBufferPotential, CL_TRUE, 0, fNWorkgroups * sizeof(CL_TYPE), fCLPotential, NULL, &event);

    event.wait();

    for (unsigned int i = 0; i < fNWorkgroups; i++)
        potential += fCLPotential[i];

    if (potential != potential)
        return fStandardSolver.Potential(aPosition);

    return potential;
}

template<class Integrator>
KThreeVector KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::ElectricField(const KPosition& aPosition) const
{
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fElectricFieldKernel,
                                                                     cl::NullRange,
                                                                     *fGlobalRange,
                                                                     *fLocalRange);

    KThreeVector eField(0., 0., 0.);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricField,
                                                                  CL_TRUE,
                                                                  0,
                                                                  fNWorkgroups * sizeof(CL_TYPE4),
                                                                  fCLElectricField);

    for (unsigned int i = 0; i < fNWorkgroups; i++) {
        eField[0] += fCLElectricField[i].s[0];
        eField[1] += fCLElectricField[i].s[1];
        eField[2] += fCLElectricField[i].s[2];
    }

    if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2])
        return fStandardSolver.ElectricField(aPosition);


    return eField;
}

template<class Integrator>
std::pair<KThreeVector, double>
KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::ElectricFieldAndPotential(const KPosition& aPosition) const
{
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fElectricFieldAndPotentialKernel,
                                                                     cl::NullRange,
                                                                     *fGlobalRange,
                                                                     *fLocalRange);

    KThreeVector eField(0., 0., 0.);
    CL_TYPE potential = 0.;

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricFieldAndPotential,
                                                                  CL_TRUE,
                                                                  0,
                                                                  fNWorkgroups * sizeof(CL_TYPE4),
                                                                  fCLElectricFieldAndPotential);

    for (unsigned int i = 0; i < fNWorkgroups; i++) {
        eField[0] += fCLElectricFieldAndPotential[i].s[0];
        eField[1] += fCLElectricFieldAndPotential[i].s[1];
        eField[2] += fCLElectricFieldAndPotential[i].s[2];
        potential += fCLElectricFieldAndPotential[i].s[3];
    }

    if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] || potential != potential)
        return fStandardSolver.ElectricFieldAndPotential(aPosition);


    return std::make_pair(eField, potential);
}

template<class Integrator>
double KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::Potential(const unsigned int* SurfaceIndexSet,
                                                                           unsigned int SetSize,
                                                                           const KPosition& aPosition) const
{
    //evaluation point
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetPotentialKernel->setArg(6, SetSize);

        //write the element ids
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        //queue kernel
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetPotentialKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);


        CL_TYPE potential = 0.;

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferPotential,
                                                                      CL_TRUE,
                                                                      0,
                                                                      n_workgroups * sizeof(CL_TYPE),
                                                                      fCLPotential,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < n_workgroups; i++) {
            potential += fCLPotential[i];
        }

        if (potential != potential) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
        }

        return potential;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
    }
}

template<class Integrator>
KThreeVector KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::ElectricField(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetElectricFieldKernel->setArg(6, SetSize);

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetElectricFieldKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);

        KThreeVector eField(0., 0., 0.);
        CL_TYPE potential = 0.;

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricField,
                                                                      CL_TRUE,
                                                                      0,
                                                                      n_workgroups * sizeof(CL_TYPE4),
                                                                      fCLElectricField,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < n_workgroups; i++) {
            eField[0] += fCLElectricField[i].s[0];
            eField[1] += fCLElectricField[i].s[1];
            eField[2] += fCLElectricField[i].s[2];
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] || potential != potential) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
        }

        return eField;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
    }
}

template<class Integrator>
std::pair<KThreeVector, double> KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::ElectricFieldAndPotential(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetElectricFieldAndPotentialKernel->setArg(6, SetSize);

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetElectricFieldAndPotentialKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);

        KThreeVector eField(0., 0., 0.);
        CL_TYPE potential = 0.;

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricFieldAndPotential,
                                                                      CL_TRUE,
                                                                      0,
                                                                      n_workgroups * sizeof(CL_TYPE4),
                                                                      fCLElectricFieldAndPotential,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < n_workgroups; i++) {
            eField[0] += fCLElectricFieldAndPotential[i].s[0];
            eField[1] += fCLElectricFieldAndPotential[i].s[1];
            eField[2] += fCLElectricFieldAndPotential[i].s[2];
            potential += fCLElectricFieldAndPotential[i].s[3];
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2]) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
        }

        return std::make_pair(eField, potential);
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
    }
}

template<class Integrator>
void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::GetReorderedSubsetIndices(
    const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const
{
    for (unsigned int i = 0; i < size; i++) {
        normal_container_ids[i] = fContainer.GetNormalIndexFromSortedIndex(sorted_container_ids[i]);
    }
}


template<class Integrator>
void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::DispatchPotential(const unsigned int* SurfaceIndexSet,
                                                                                 unsigned int SetSize,
                                                                                 const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        //evaluation point
        CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetPotentialKernel->setArg(6, SetSize);

        //write the element ids
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

        //queue kernel
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetPotentialKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);
        fCallDevice = true;
        return;
    }
    else {
        fCallDevice = false;
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetPotential = fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
    }
}

template<class Integrator>
void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::DispatchElectricField(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetElectricFieldKernel->setArg(6, SetSize);

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetElectricFieldKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);

        fCallDevice = true;
        return;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetField = fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
        fCallDevice = false;
    }
}

template<class Integrator>
void KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::DispatchElectricFieldAndPotential(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if (SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize) {
        CL_TYPE P[3] = {aPosition[0], aPosition[1], aPosition[2]};

        //write point
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferP, CL_TRUE, 0, 3 * sizeof(CL_TYPE), P);

        //write number of elements
        fSubsetElectricFieldAndPotentialKernel->setArg(6, SetSize);

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferElementIdentities,
                                                                       CL_TRUE,
                                                                       0,
                                                                       SetSize * sizeof(unsigned int),
                                                                       SurfaceIndexSet);

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        unsigned int n_workgroups = n_global / fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSubsetElectricFieldAndPotentialKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);

        fCallDevice = true;
        return;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetFieldAndPotential =
            fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
        fCallDevice = false;
    }
}

template<class Integrator> double KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::RetrievePotential() const
{
    if (fCallDevice) {
        CL_TYPE potential = 0.;

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferPotential,
                                                                      CL_TRUE,
                                                                      0,
                                                                      fCachedNWorkgroups * sizeof(CL_TYPE),
                                                                      fCLPotential,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < fCachedNWorkgroups; i++) {
            potential += fCLPotential[i];
        }

        if (potential != potential) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.Potential(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return potential;
    }
    else {
        return fCachedSubsetPotential;
    }
}

template<class Integrator>
KThreeVector KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::RetrieveElectricField() const
{
    if (fCallDevice) {
        KThreeVector eField(0., 0., 0.);

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricField,
                                                                      CL_TRUE,
                                                                      0,
                                                                      fCachedNWorkgroups * sizeof(CL_TYPE4),
                                                                      fCLElectricField,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < fCachedNWorkgroups; i++) {
            eField[0] += fCLElectricField[i].s[0];
            eField[1] += fCLElectricField[i].s[1];
            eField[2] += fCLElectricField[i].s[2];
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2]) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.ElectricField(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return eField;
    }
    else {
        return fCachedSubsetField;
    }
}

template<class Integrator>
std::pair<KThreeVector, double>
KIntegratingFieldSolver<Integrator, ElectrostaticOpenCL>::RetrieveElectricFieldAndPotential() const
{
    if (fCallDevice) {
        KThreeVector eField(0., 0., 0.);
        CL_TYPE potential = 0.;

        cl::Event event;
        KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferElectricFieldAndPotential,
                                                                      CL_TRUE,
                                                                      0,
                                                                      fCachedNWorkgroups * sizeof(CL_TYPE4),
                                                                      fCLElectricFieldAndPotential,
                                                                      NULL,
                                                                      &event);

        event.wait();

        for (unsigned int i = 0; i < fCachedNWorkgroups; i++) {
            eField[0] += fCLElectricFieldAndPotential[i].s[0];
            eField[1] += fCLElectricFieldAndPotential[i].s[1];
            eField[2] += fCLElectricFieldAndPotential[i].s[2];
            potential += fCLElectricFieldAndPotential[i].s[3];
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] || potential != potential) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return std::make_pair(eField, potential);
    }
    else {
        return fCachedSubsetFieldAndPotential;
    }
}

}  // namespace KEMField

#endif /* KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
