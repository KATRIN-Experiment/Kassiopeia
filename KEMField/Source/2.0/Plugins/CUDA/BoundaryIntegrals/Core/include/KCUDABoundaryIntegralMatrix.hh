#ifndef KCUDABOUNDARYINTEGRALMATRIX_DEF
#define KCUDABOUNDARYINTEGRALMATRIX_DEF

#include <fstream>
#include <sstream>

#include "kEMField_LinearAlgebra_kernel.cuh"

#include "KCUDAAction.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KCUDASurfaceContainer.hh"
#include "KCUDABoundaryIntegrator.hh"

namespace KEMField
{
  template <class BasisPolicy>
  class KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> > : public KSquareMatrix<typename BasisPolicy::ValueType>, public KCUDAAction
  {
  public:
    typedef typename BasisPolicy::ValueType ValueType;
    friend class KCUDASurfaceContainer;

    KBoundaryIntegralMatrix(KCUDASurfaceContainer& c, KCUDABoundaryIntegrator<BasisPolicy>& integrator);

    ~KBoundaryIntegralMatrix();

    unsigned int Dimension() const { return fDimension; }

    const ValueType& operator()(unsigned int i,unsigned int j) const;

    void SetNLocal(int nLocal) const { fNLocal = nLocal; }
    int GetNLocal() const { return fNLocal; }

    KCUDABoundaryIntegrator<BasisPolicy>& GetIntegrator() const { return fIntegrator; }

  private:
    KCUDASurfaceContainer& fContainer;
    KCUDABoundaryIntegrator<BasisPolicy>& fIntegrator;

    const unsigned int fDimension;

    void ConstructCUDAKernels() const;
    void AssignDeviceMemory() const;

    mutable int fNLocal;

    mutable int *fDeviceIJ;
    mutable double *fDeviceValue;

    mutable ValueType fValue;
  };

  template <class BasisPolicy>
  KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> >::KBoundaryIntegralMatrix(KCUDASurfaceContainer& c,KCUDABoundaryIntegrator<BasisPolicy>& integrator) :
    KSquareMatrix<ValueType>(),
    KCUDAAction(c),
    fContainer(c),
    fIntegrator(integrator),
    fDimension(c.size()*BasisPolicy::Dimension),
    fNLocal(-1),
    fDeviceIJ(NULL),
    fDeviceValue(NULL)
  {

  }

  template <class BasisPolicy>
  KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> >::~KBoundaryIntegralMatrix()
  {
    if( fDeviceIJ ) cudaFree( fDeviceIJ );
    if( fDeviceValue ) cudaFree( fDeviceValue );
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> >::operator()(unsigned int i,unsigned int j) const
  {
    int ij[2];
    ij[0] = static_cast<int>(i);
    ij[1] = static_cast<int>(j);

    cudaMemcpy( fDeviceIJ, ij, 2*sizeof(int), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, GetMatrixElementKernel, 0, 0);
        std::cout << "[GetMatrixElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[GetMatrixElementKernel] Set block size: " << local << std::endl;
#endif

    GetMatrixElementKernel <<<global, local>>> (
            fDeviceIJ,
            fContainer.GetBoundaryInfo(),
            fContainer.GetBoundaryData(),
            fContainer.GetShapeInfo(),
            fContainer.GetShapeData(),
            fDeviceValue );

    CU_TYPE value;
    cudaMemcpy( &value, fDeviceValue, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );
    fValue = value;
    return fValue;
  }


  template <class BasisPolicy>
  void KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> >::ConstructCUDAKernels() const
  {
    // device management

    // define fNLocal
    if( fNLocal == -1 ) fNLocal = 384;

    fData.SetMinimumWorkgroupSizeForKernels(fNLocal);

    // Create memory buffers
    cudaMalloc( (void**) &fDeviceIJ, 2*sizeof(int) );
    cudaMalloc( (void**) &fDeviceValue, sizeof(CU_TYPE) );
  }

  template <class BasisPolicy>
  void KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<BasisPolicy> >::AssignDeviceMemory() const
  {

  }
}

#endif /* KCUDABOUNDARYINTEGRALMATRIX_DEF */
