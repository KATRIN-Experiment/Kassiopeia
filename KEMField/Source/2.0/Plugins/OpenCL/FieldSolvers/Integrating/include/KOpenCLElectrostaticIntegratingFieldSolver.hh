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

    //use this constructor when sub-set solving is to be used
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
    double Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    KEMThreeVector ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;

    //these methods allow us to dispatch a calculation to the GPU and retrieve the values later
    //this is useful so that we can do other work while waiting for the results
    void DispatchPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    void DispatchElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    double RetrievePotential() const;
    KEMThreeVector RetrieveElectricField() const;

    ////////////////////////////////////////////////////////////////////////////

    std::string GetOpenCLFlags() const { return fOpenCLFlags; }

  private:

    void GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const;

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
    mutable unsigned int* fSubsetIdentities;
    mutable unsigned int* fReorderedSubsetIdentities;

    //for disptached and later collected subset potential/field
    mutable unsigned int fCachedNGlobal;
    mutable unsigned int fCachedNDummy;
    mutable unsigned int fCachedNWorkgroups;
    mutable double fCachedSubsetPotential;
    mutable KEMThreeVector fCachedSubsetField;

    mutable unsigned int fCachedSubsetSize;
    mutable const unsigned int* fCachedSurfaceIndexSet;
    mutable KPosition fCachedPosition;

    mutable bool fCallDevice;

  };
}

#endif /* KOPENCLELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
