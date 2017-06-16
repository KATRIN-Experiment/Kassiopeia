#ifndef KCUDABOUNDARYINTEGRALVECTOR_DEF
#define KCUDABOUNDARYINTEGRALVECTOR_DEF

#include "kEMField_LinearAlgebra_kernel.cuh"

#include "KCUDAAction.hh"
#include "KBoundaryIntegralVector.hh"
#include "KCUDASurfaceContainer.hh"
#include "KCUDABoundaryIntegrator.hh"

namespace KEMField
{
  template <class BasisPolicy>
  class KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> > : public KVector<typename BasisPolicy::ValueType>, public KCUDAAction
  {
  public:
    typedef typename BasisPolicy::ValueType ValueType;
    friend class KCUDASurfaceContainer;

    KBoundaryIntegralVector(KCUDASurfaceContainer& c, KCUDABoundaryIntegrator<BasisPolicy>& integrator);

    ~KBoundaryIntegralVector();

    unsigned int Dimension() const { return fDimension; }

    const ValueType& operator()(unsigned int i) const;

    const ValueType& InfinityNorm() const;

    void SetNLocal(int nLocal) const { fNLocal = nLocal; }

    int GetNLocal() const { return fNLocal; }

  private:
    // We disable this method by making it private.
    virtual ValueType& operator[](unsigned int ) { static ValueType v; return v; }

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
  KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::KBoundaryIntegralVector(KCUDASurfaceContainer& c,KCUDABoundaryIntegrator<BasisPolicy>& integrator) :
    KVector<ValueType>(),
    KCUDAAction(c),
    fContainer(c),
    fIntegrator(integrator),
    fDimension(c.size()*BasisPolicy::Dimension),
    fNLocal(-1),
    fDeviceI(NULL),
    fDeviceValue(NULL)
  {

  }

  template <class BasisPolicy>
  KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::~KBoundaryIntegralVector()
  {
    if (fDeviceI) cudaFree(fDeviceI);
    if (fDeviceValue) cudaFree(fDeviceValue);
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::operator()(unsigned int i) const
  {
    int i_[1];
    i_[0] = static_cast<int>(i);
    cudaMemcpy( fDeviceI, i_, sizeof(int), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetVectorElementKernel, 0, 0);
        std::cout << "[GetVectorElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetVectorElementKernel] Set block size: " << local << std::endl;
#endif

    GetVectorElementKernel <<<global,local>>> (
            fDeviceI,
            fContainer.GetBoundaryInfo(),
            fContainer.GetBoundaryData(),
            fDeviceValue );

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::InfinityNorm() const
  {
    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetMaximumVectorElementKernel, 0, 0);
        std::cout << "[GetMaximumVectorElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetMaximumVectorElementKernel] Set block size: " << local << std::endl;
#endif

    GetMaximumVectorElementKernel <<<global,local>>> (
            fContainer.GetBoundaryInfo(),
            fContainer.GetBoundaryData(),
            fDeviceValue );

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  void KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::ConstructCUDAKernels() const
  {
    // define fNLocal
    if( fNLocal == -1 ) fNLocal = 384;

    // Create memory buffers
    cudaMalloc( (void**) &fDeviceI, sizeof(int) );
    cudaMalloc( (void**) &fDeviceValue, sizeof(CU_TYPE) );
  }

  template <class BasisPolicy>
  void KBoundaryIntegralVector<KCUDABoundaryIntegrator<BasisPolicy> >::AssignDeviceMemory() const
  {

  }
}

#endif /* KCUDABOUNDARYINTEGRALVECTOR_DEF */
