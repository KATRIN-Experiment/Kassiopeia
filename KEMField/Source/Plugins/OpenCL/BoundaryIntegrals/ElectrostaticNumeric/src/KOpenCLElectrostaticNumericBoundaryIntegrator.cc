#include "KOpenCLElectrostaticNumericBoundaryIntegrator.hh"

#include "KOpenCLSurfaceContainer.hh"
#include "KSurfaceTypes.hh"

#include <fstream>
#include <iostream>
#include <sstream>

namespace KEMField
{
KOpenCLElectrostaticNumericBoundaryIntegrator::KOpenCLElectrostaticNumericBoundaryIntegrator(
    KOpenCLSurfaceContainer& c) :
    KOpenCLBoundaryIntegrator<KElectrostaticBasis>(c),
    fPhiKernel(NULL),
    fEFieldKernel(NULL),
    fEFieldAndPhiKernel(NULL),
    fBufferPhi(NULL),
    fBufferEField(NULL),
    fBufferEFieldAndPhi(NULL)
{
    std::stringstream options;
    options << GetOpenCLFlags() << " -DKEMFIELD_INTEGRATORFILE_CL=<" << OpenCLFile() << ">";
    options << " -DKEMFIELD_OCLFASTRWG=" << KEMFIELD_FASTRWG_VALUE; /* variable defined via cmake */
    fOpenCLFlags = options.str();
    ConstructOpenCLKernels();
    AssignBuffers();
}

KOpenCLElectrostaticNumericBoundaryIntegrator::~KOpenCLElectrostaticNumericBoundaryIntegrator()
{
    if (fPhiKernel)
        delete fPhiKernel;
    if (fEFieldKernel)
        delete fEFieldKernel;
    if (fEFieldAndPhiKernel)
        delete fEFieldAndPhiKernel;

    if (fBufferPhi)
        delete fBufferPhi;
    if (fBufferEField)
        delete fBufferEField;
    if (fBufferEFieldAndPhi)
        delete fBufferEFieldAndPhi;
}

void KOpenCLElectrostaticNumericBoundaryIntegrator::BoundaryVisitor::Visit(KDirichletBoundary& boundary)
{
    fIsDirichlet = true;
    fPrefactor = 1.;
    fBoundaryValue = static_cast<DirichletBoundary&>(boundary).GetBoundaryValue();
}

void KOpenCLElectrostaticNumericBoundaryIntegrator::BoundaryVisitor::Visit(KNeumannBoundary& boundary)
{
    fIsDirichlet = false;
    fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()) /
                  (1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()));
    fBoundaryValue = 0.;
}

void KOpenCLElectrostaticNumericBoundaryIntegrator::BasisVisitor::Visit(KElectrostaticBasis& basis)
{
    fBasisValue = &(basis.GetSolution());
}

KElectrostaticBasis::ValueType
KOpenCLElectrostaticNumericBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source, KSurfacePrimitive* target,
                                                                unsigned int)
{
    fTarget = target;
    target->Accept(fBoundaryVisitor);
    source->Accept(*this);
    return fValue;
}

KElectrostaticBasis::ValueType KOpenCLElectrostaticNumericBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,
                                                                                            unsigned int)
{
    surface->Accept(fBoundaryVisitor);
    return fBoundaryVisitor.GetBoundaryValue();
}

KElectrostaticBasis::ValueType& KOpenCLElectrostaticNumericBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface,
                                                                                          unsigned int)
{
    surface->Accept(fBasisVisitor);
    return fBasisVisitor.GetBasisValue();
}

void KOpenCLElectrostaticNumericBoundaryIntegrator::ConstructOpenCLKernels() const
{
    // Constructs an OpenCL program for computing the electric potential due to
    // a charged rectangle/triangle/wire, and builds a queue for it.

    KOpenCLBoundaryIntegrator<KElectrostaticBasis>::ConstructOpenCLKernels();

    // Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath()
           << "/kEMField_ElectrostaticNumericBoundaryIntegrals_kernel.cl";

    // Read kernel source from file
    std::string sourceCode;
    std::ifstream sourceFile(clFile.str().c_str());

    sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    // Make program of the source code in the context
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source);

    std::stringstream options;
    options << GetOpenCLFlags() << fData.GetOpenCLFlags();

    // Build program for these specific devices
    try {
        // use only target device!
        CL_VECTOR_TYPE<cl::Device> devices;
        devices.push_back(KOpenCLInterface::GetInstance()->GetDevice());
        program.build(devices, options.str().c_str());
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::stringstream s;
        s << "There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:" << std::endl;
        s << error.what() << "(" << error.err() << ")" << std::endl;
        s << "Build Status: "
          << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice()) << std::endl;
        s << "Build Options:\t"
          << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice()) << std::endl;
        s << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())
          << std::endl;
        std::cout << s.str() << std::endl;
    }

#ifdef DEBUG_OPENCL_COMPILER_OUTPUT
    std::stringstream s;
    s << "Build Log for OpenCL " << clFile.str() << " :\t ";
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

    // Make kernel
    fPhiKernel = new cl::Kernel(program, "Potential");
    fEFieldKernel = new cl::Kernel(program, "ElectricField");
    fEFieldAndPhiKernel = new cl::Kernel(program, "ElectricFieldAndPotential");

    // Create memory buffers
    fBufferP = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, 3 * sizeof(CL_TYPE));
    fBufferShapeInfo =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sizeof(cl_short));
    fBufferShapeData = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_READ_ONLY,
                                      // Hard-coded arbitrary maximum shape limit
                                      20 * sizeof(CL_TYPE));
    fBufferPhi = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, sizeof(CL_TYPE));
    fBufferEField = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, sizeof(CL_TYPE4));
    fBufferEFieldAndPhi =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, sizeof(CL_TYPE4));
}

void KOpenCLElectrostaticNumericBoundaryIntegrator::AssignBuffers() const
{
    KOpenCLBoundaryIntegrator<KElectrostaticBasis>::AssignBuffers();

    //std::cout << "Numerical Potential       - Workgroup Size: " << fPhiKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice())) << std::endl;

    // Set arguments to kernel
    fPhiKernel->setArg(0, *fBufferP);
    fPhiKernel->setArg(1, *fBufferShapeInfo);
    fPhiKernel->setArg(2, *fBufferShapeData);
    fPhiKernel->setArg(3, *fBufferPhi);

    //std::cout << "Numerical Electric Field  - Workgroup Size: " << fEFieldKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice())) << std::endl;

    fEFieldKernel->setArg(0, *fBufferP);
    fEFieldKernel->setArg(1, *fBufferShapeInfo);
    fEFieldKernel->setArg(2, *fBufferShapeData);
    fEFieldKernel->setArg(3, *fBufferEField);

    //std::cout << "Numerical EField and Pot  - Workgroup Size: " << fEFieldAndPhiKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice())) << std::endl;

    fEFieldAndPhiKernel->setArg(0, *fBufferP);
    fEFieldAndPhiKernel->setArg(1, *fBufferShapeInfo);
    fEFieldAndPhiKernel->setArg(2, *fBufferShapeData);
    fEFieldAndPhiKernel->setArg(3, *fBufferEFieldAndPhi);
}
}  // namespace KEMField
