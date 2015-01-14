#ifndef KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF
#define KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF

#include "KOpenCLAction.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

namespace KEMField
{
  template <class Integrator>
  class KIntegratingFieldSolver;

  template <>
  class KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator> :
    public KOpenCLAction
  {
  public:
    typedef KOpenCLElectrostaticBoundaryIntegrator::Basis Basis;

    KIntegratingFieldSolver(KOpenCLSurfaceContainer& container,
			    KOpenCLElectrostaticBoundaryIntegrator& integrator);

    //use this constructo when sub-set solving is to be used
    KIntegratingFieldSolver(KOpenCLSurfaceContainer& container,
			    KOpenCLElectrostaticBoundaryIntegrator& integrator,
                unsigned int max_subset_size,
                unsigned int min_subset_size = 16);

    virtual ~KIntegratingFieldSolver();

    void ConstructOpenCLKernels() const;
    void AssignBuffers() const;

    double Potential(const KPosition& aPosition) const;
    KEMThreeVector ElectricField(const KPosition& aPosition) const;

    ////////////////////////////////////////////////////////////////////////////
    //sub-set potential/field calls
    double Potential(const std::vector<unsigned int>* SurfaceIndexSet, const KPosition& aPosition) const;
    KEMThreeVector ElectricField(const std::vector<unsigned int>* SurfaceIndexSet, const KPosition& aPosition) const;
    ////////////////////////////////////////////////////////////////////////////

    std::string GetOpenCLFlags() const { return fOpenCLFlags; }

  private:
    KOpenCLSurfaceContainer& fContainer;

    KElectrostaticBoundaryIntegrator fStandardIntegrator;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator> fStandardSolver;

    mutable std::string fOpenCLFlags;

    mutable cl::Kernel* fPotentialKernel;
    mutable cl::Kernel* fElectricFieldKernel;

    mutable cl::Buffer *fBufferP;
    mutable cl::Buffer *fBufferPotential;
    mutable cl::Buffer *fBufferElectricField;

    mutable cl::NDRange* fGlobalRange;
    mutable cl::NDRange* fLocalRange;

    mutable CL_TYPE* fCLPotential;
    mutable CL_TYPE4* fCLElectricField;

    mutable unsigned int fNGlobal;
    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    ////////////////////////////////////////////////////////////////////////////

    //sub-set kernel and buffers
    unsigned int fMaxSubsetSize;
    unsigned int fMinSubsetSize;
    mutable cl::Kernel* fSubsetPotentialKernel;
    mutable cl::Kernel* fSubsetElectricFieldKernel;
    mutable cl::Buffer* fBufferElementIdentities;

  };
}

#endif /* KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
