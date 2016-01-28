#ifndef KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF
#define KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF

#include "KCUDAAction.hh"
#include "KCUDASurfaceContainer.hh"
#include "KCUDAElectrostaticBoundaryIntegrator.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "kEMField_ElectrostaticIntegratingFieldSolver_kernel.cuh"

namespace KEMField
{
  template <class Integrator>
  class KIntegratingFieldSolver;

  template <>
  class KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator> :
    public KCUDAAction
  {
  public:
    typedef KCUDAElectrostaticBoundaryIntegrator::Basis Basis;

    KIntegratingFieldSolver(KCUDASurfaceContainer& container,
			    KCUDAElectrostaticBoundaryIntegrator& integrator);

    //use this constructor when sub-set solving is to be used
    KIntegratingFieldSolver(KCUDASurfaceContainer& container,
			    KCUDAElectrostaticBoundaryIntegrator& integrator,
                unsigned int max_subset_size,
                unsigned int min_subset_size = 16);

    virtual ~KIntegratingFieldSolver();

    void ConstructCUDAKernels() const;
    void AssignDeviceMemory() const;

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

  private:

    void GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const;

    KCUDASurfaceContainer& fContainer;

    KElectrostaticBoundaryIntegrator fStandardIntegrator;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator> fStandardSolver;

    mutable CU_TYPE *fDeviceP;
    mutable CU_TYPE *fDevicePotential;
    mutable CU_TYPE4 *fDeviceElectricField;

    mutable CU_TYPE* fHostPotential;
    mutable CU_TYPE4* fHostElectricField;

    mutable unsigned int fNGlobal;
    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    ////////////////////////////////////////////////////////////////////////////

    //sub-set kernel and buffers
    unsigned int fMaxSubsetSize;
    unsigned int fMinSubsetSize;
    mutable unsigned int* fDeviceElementIdentities;
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

#endif /* KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
