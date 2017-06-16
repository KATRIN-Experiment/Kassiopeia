#ifndef KCUDABOUNDARYINTEGRALSOLUTIONVECTOR_DEF
#define KCUDABOUNDARYINTEGRALSOLUTIONVECTOR_DEF

#include "kEMField_LinearAlgebra_kernel.cuh"

#include "KCUDAAction.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KCUDASurfaceContainer.hh"
#include "KCUDABoundaryIntegrator.hh"

namespace KEMField
{
  template <class BasisPolicy>
  class KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> > : public KVector<typename BasisPolicy::ValueType>, public KCUDAAction
  {
  public:
    typedef typename BasisPolicy::ValueType ValueType;
    friend class KCUDASurfaceContainer;

    KBoundaryIntegralSolutionVector(KCUDASurfaceContainer& c, KCUDABoundaryIntegrator<BasisPolicy>& integrator);

    ~KBoundaryIntegralSolutionVector();

    unsigned int Dimension() const { return fDimension; }

    const ValueType& operator()(unsigned int i) const;
    // Currently, this method does not return a modifiable value
    ValueType& operator[](unsigned int i);

    const ValueType& InfinityNorm() const;

    void SetNLocal(int nLocal) const { fNLocal = nLocal; }

    int GetNLocal() const { return fNLocal; }

  private:
    KCUDASurfaceContainer& fContainer;
    KCUDABoundaryIntegrator<BasisPolicy>& fIntegrator;

    const unsigned int fDimension;

    void ConstructCUDAKernels() const;
    void AssignDeviceMemory() const;

    mutable int fNLocal;

    mutable int *fDeviceI;
    mutable CU_TYPE *fDeviceValue;

    mutable ValueType fValue;
  };

  template <class BasisPolicy>
  KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::KBoundaryIntegralSolutionVector(KCUDASurfaceContainer& c,KCUDABoundaryIntegrator<BasisPolicy>& integrator) :
    KVector<ValueType>(),
    KCUDAAction(c),
    fContainer(c),
    fIntegrator(integrator),
    // TO DO: add mult. factors
    fDimension(c.size()),
    fNLocal(-1),
    fDeviceI(NULL),
    fDeviceValue(NULL)
  {

  }

  template <class BasisPolicy>
  KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::~KBoundaryIntegralSolutionVector()
  {
    if( fDeviceI ) cudaFree( fDeviceI );
    if( fDeviceValue ) cudaFree( fDeviceValue );
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::operator()(unsigned int i) const
  {
    int i_[1];
    i_[0] = static_cast<int>(i);
    cudaMemcpy( fDeviceI, i_, sizeof(int), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetSolutionVectorElementKernel, 0, 0);
        std::cout << "[GetSolutionVectorElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetSolutionVectorElementKernel] Set block size: " << local << std::endl;
#endif

    GetSolutionVectorElementKernel <<<global,local>>> (
        fDeviceI,
        fContainer.GetBasisData(),
        fDeviceValue );

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::operator[](unsigned int i)
  {
    int i_[1];
    i_[0] = static_cast<int>(i);

    cudaMemcpy( fDeviceI, i_, sizeof(int), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetSolutionVectorElementKernel, 0, 0);
        std::cout << "[GetSolutionVectorElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetSolutionVectorElementKernel] Set block size: " << local << std::endl;
#endif

    GetSolutionVectorElementKernel <<<global,local>>> (
        fDeviceI,
        fContainer.GetBasisData(),
        fDeviceValue);

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::InfinityNorm() const
  {
    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetMaximumSolutionVectorElementKernel, 0, 0);
        std::cout << "[GetMaximumSolutionVectorElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetMaximumSolutionVectorElementKernel] Set block size: " << local << std::endl;
#endif

    GetMaximumSolutionVectorElementKernel <<<global,local>>> (
        fContainer.GetBoundaryInfo(),
        fContainer.GetBasisData(),
        fDeviceValue );

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  void KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::ConstructCUDAKernels() const
  {
    // define fNLocal
    if( fNLocal == -1 ) fNLocal = 384;

    // Create memory buffers
    cudaMalloc( (void**) &fDeviceI, sizeof(int) );
    cudaMalloc( (void**) &fDeviceValue, sizeof(CU_TYPE) );
  }

  template <class BasisPolicy>
  void KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<BasisPolicy> >::AssignDeviceMemory() const
  {

  }
}

#endif /* KCUDABOUNDARYINTEGRALSOLUTIONVECTOR_DEF */
