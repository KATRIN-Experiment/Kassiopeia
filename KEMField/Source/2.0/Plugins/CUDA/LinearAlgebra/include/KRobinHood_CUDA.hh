#ifndef KROBINHOOD_CUDA_DEF
#define KROBINHOOD_CUDA_DEF

#include "KRobinHood.hh"

#include <limits.h>

#include "kEMField_RobinHood_kernel.cuh"
#include "KCUDAAction.hh"
#include "KCUDABoundaryIntegralMatrix.hh"

namespace KEMField
{
  template <typename ValueType>
  class KRobinHood_CUDA : public KCUDAAction
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KRobinHood_CUDA(const Matrix& A,Vector& x,const Vector& b);
    ~KRobinHood_CUDA();

    void ConstructCUDAKernels() const;
    void AssignDeviceMemory() const;

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void CompleteResidualNormalization(double& residualNorm);
    void IdentifyLargestResidualElement();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData();
    void Finalize();

    unsigned int Dimension() const { return fB.Dimension(); }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    ///////////////////////////////

    // device data for CUDA
    mutable CU_TYPE *fDeviceResidual;
    mutable CU_TYPE *fDeviceB_iterative;
    mutable CU_TYPE *fDeviceCorrection;
    mutable int     *fDevicePartialMaxResidualIndex;
    mutable int     *fDeviceMaxResidualIndex;
    mutable CU_TYPE *fDevicePartialResidualNorm;
    mutable CU_TYPE *fDeviceResidualNorm;
    mutable int     *fDeviceNWarps;
    mutable int     *fDeviceCounter;

    ////////////////

    // host data for CUDA
    mutable CU_TYPE *fHostResidual;
    mutable CU_TYPE *fHostB_iterative;
    mutable CU_TYPE *fHostCorrection;
    mutable int     *fHostPartialMaxResidualIndex;
    mutable int     *fHostMaxResidualIndex;
    mutable CU_TYPE *fHostPartialResidualNorm;
    mutable CU_TYPE *fHostResidualNorm;
    mutable int     *fHostNWarps;
    mutable int     *fHostCounter;

    mutable bool fReadResidual;
  };

  template <typename ValueType>
  KRobinHood_CUDA<ValueType>::KRobinHood_CUDA(const Matrix& A, Vector& x, const Vector& b) :
      KCUDAAction((dynamic_cast<const KCUDAAction&>(A)).GetData()),
      fA(A), fX(x), fB(b),
      fDeviceResidual(NULL),
      fDeviceB_iterative(NULL),
      fDeviceCorrection(NULL),
      fDevicePartialMaxResidualIndex(NULL),
      fDeviceMaxResidualIndex(NULL),
      fDevicePartialResidualNorm(NULL),
      fDeviceResidualNorm(NULL),
      fDeviceNWarps(NULL),
      fDeviceCounter(NULL),
      fHostResidual(NULL),
      fHostB_iterative(NULL),
      fHostCorrection(NULL),
      fHostPartialMaxResidualIndex(NULL),
      fHostMaxResidualIndex(NULL),
      fHostPartialResidualNorm(NULL),
      fHostResidualNorm(NULL),
      fHostNWarps(NULL),
      fHostCounter(NULL),
      fReadResidual(false)
  {
      KCUDAAction::Initialize();
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::ConstructCUDAKernels() const
  {
     fNLocal = 384;
     fData.SetMinimumWorkgroupSizeForKernels(fNLocal);
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::AssignDeviceMemory() const
  {
    fNLocal = fData.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fData.GetNBufferedElements()/fNLocal;

    KCUDASurfaceContainer& container = dynamic_cast<KCUDASurfaceContainer&>(fData);

    fHostB_iterative = new CU_TYPE[fData.GetNBufferedElements()];

    int maxNumWarps = fData.GetNBufferedElements()/fNLocal;

    fHostNWarps = new int[1];
    fHostNWarps[0] = fData.GetNBufferedElements()/fNLocal;
    fHostCounter = new int[1];
    fHostCounter[0] = fData.GetNBufferedElements();

    fHostResidual = new CU_TYPE[fData.GetNBufferedElements()];
    fHostCorrection = new CU_TYPE[1];
    fHostPartialMaxResidualIndex = new int[maxNumWarps];
    fHostMaxResidualIndex = new int[1];
    fHostPartialResidualNorm = new CU_TYPE[maxNumWarps];
    fHostResidualNorm = new CU_TYPE[1];

    for (unsigned int i=0;i<fData.GetNBufferedElements();i++) {
      if (i<container.size())
        fHostB_iterative[i] = 0.;
      else
        fHostB_iterative[i] = 1.e30;

      fHostResidual[i] = 0.;
    }

    fHostCorrection[0] = 0.;
    fHostMaxResidualIndex[0] = -1;
    fHostResidualNorm[0] = 0.;

    // Memory allocation on device

    cudaMalloc((void**) &fDeviceResidual, container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceB_iterative, container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceCorrection, sizeof(CU_TYPE));
    cudaMalloc((void**) &fDevicePartialMaxResidualIndex, maxNumWarps * sizeof(int));
    cudaMalloc((void**) &fDeviceMaxResidualIndex, sizeof(int));
    cudaMalloc((void**) &fDevicePartialResidualNorm, maxNumWarps * sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceResidualNorm, sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceNWarps, sizeof(int));
    cudaMalloc((void**) &fDeviceCounter, sizeof(int));

    // Copy lists to device memory

    cudaMemcpy(fDeviceResidual, fHostResidual, container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceB_iterative, fHostB_iterative, container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceCorrection, fHostCorrection, sizeof(CU_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(fDevicePartialMaxResidualIndex, fHostPartialMaxResidualIndex, maxNumWarps * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceMaxResidualIndex, fHostMaxResidualIndex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fDevicePartialResidualNorm, fHostPartialResidualNorm, maxNumWarps * sizeof(CU_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceResidualNorm, fHostResidualNorm, sizeof(CU_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceNWarps, fHostNWarps, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fDeviceCounter, fHostCounter, sizeof(int), cudaMemcpyHostToDevice);
  }

  template <typename ValueType>
  KRobinHood_CUDA<ValueType>::~KRobinHood_CUDA()
  {
    if (fDeviceResidual) cudaFree(fDeviceResidual);
    if (fDeviceB_iterative) cudaFree(fDeviceB_iterative);
    if (fDeviceCorrection) cudaFree(fDeviceCorrection);
    if (fDevicePartialMaxResidualIndex) cudaFree(fDevicePartialMaxResidualIndex);
    if (fDeviceMaxResidualIndex) cudaFree(fDeviceMaxResidualIndex);
    if (fDevicePartialResidualNorm) cudaFree(fDevicePartialResidualNorm);
    if (fDeviceResidualNorm) cudaFree(fDeviceResidualNorm);
    if (fDeviceNWarps) cudaFree(fDeviceNWarps);
    if (fDeviceCounter) cudaFree(fDeviceCounter);

    if (fHostResidual) delete fHostResidual;
    if (fHostB_iterative) delete fHostB_iterative;
    if (fHostCorrection) delete fHostCorrection;
    if (fHostPartialMaxResidualIndex) delete fHostPartialMaxResidualIndex;
    if (fHostMaxResidualIndex) delete fHostMaxResidualIndex;
    if (fHostPartialResidualNorm) delete fHostPartialResidualNorm;
    if (fHostResidualNorm) delete fHostResidualNorm;
    if (fHostNWarps) delete fHostNWarps;
    if (fHostCounter) delete fHostCounter;
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::Initialize()
  {
    if( !fReadResidual ) {
      if( fX.InfinityNorm()>1.e-16 ) {

#ifdef KEMFIELD_OCCUPANCYAPI
		  int blockSize = 0;   // The launch configurator returned block size
		  int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
		  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, InitializeVectorApproximationKernel, 0, 0);
		  std::cout << "[InitializeVectorApproximationKernel] Suggested block size: " << blockSize << std::endl;
		  std::cout << "[InitializeVectorApproximationKernel] Set block size: " << fNLocal << std::endl;
#endif

    	  InitializeVectorApproximationKernel <<<fNWorkgroups,fNLocal>>> (
					dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
					dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryData(),
					dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeInfo(),
					dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeData(),
					dynamic_cast<KCUDASurfaceContainer&>(fData).GetBasisData(),
					fDeviceResidual );

          cudaDeviceSynchronize();
      }
    }
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::FindResidual()
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, FindResidualKernel, 0, 0);
        std::cout << "[FindResidualKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[FindResidualKernel] Set block size: " << fNLocal << std::endl;
#endif

      FindResidualKernel <<<fNWorkgroups, fNLocal>>> (
              fDeviceResidual,
              dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
              dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryData(),
              fDeviceB_iterative,
              fDeviceCounter );
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::FindResidualNorm(double& residualNorm)
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, FindResidualNormKernel, 0, 0);
        std::cout << "[FindResidualNormKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[FindResidualNormKernel] Set block size: " << "1" << std::endl;
#endif

    FindResidualNormKernel <<<1,1>>> (
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
        fDeviceResidual,
        fDeviceResidualNorm);

    CompleteResidualNormalization(residualNorm);
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::CompleteResidualNormalization(double& residualNorm)
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, CompleteResidualNormalizationKernel, 0, 0);
        std::cout << "[CompleteResidualNormalizationKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[CompleteResidualNormalizationKernel] Set block size: " << "1" << std::endl;
#endif

    CompleteResidualNormalizationKernel <<<1, 1>>> (
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryData(),
        fDeviceResidualNorm);

    cudaMemcpy( fHostResidualNorm, fDeviceResidualNorm, sizeof(CU_TYPE), cudaMemcpyDeviceToHost);

    residualNorm = fHostResidualNorm[0];
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::IdentifyLargestResidualElement()
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, IdentifyLargestResidualElementKernel, 0, 0);
        std::cout << "[IdentifyLargestResidualElementKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[IdentifyLargestResidualElementKernel] Set block size: " << fNLocal << std::endl;
#endif

    IdentifyLargestResidualElementKernel <<<fNWorkgroups, fNLocal, fNLocal*sizeof(int)>>> (
    		fDeviceResidual,
    		fDevicePartialMaxResidualIndex );

#ifdef KEMFIELD_OCCUPANCYAPI
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, CompleteLargestResidualIdentificationKernel, 0, 0);
        std::cout << "[CompleteLargestResidualIdentificationKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[CompleteLargestResidualIdentificationKernel] Set block size: " << "1" << std::endl;
#endif

    CompleteLargestResidualIdentificationKernel <<<1, 1>>> (
        fDeviceResidual,
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
        fDevicePartialMaxResidualIndex,
        fDeviceMaxResidualIndex,
        fDeviceNWarps);
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::ComputeCorrection()
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ComputeCorrectionKernel, 0, 0);
        std::cout << "[ComputeCorrectionKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[ComputeCorrectionKernel] Set block size: " << "1" << std::endl;
#endif

    ComputeCorrectionKernel <<<1, 1>>> (
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeInfo(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeData(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryData(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBasisData(),
        fDeviceB_iterative,
        fDeviceCorrection,
        fDeviceMaxResidualIndex,
        fDeviceCounter);
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::UpdateSolutionApproximation()
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, UpdateSolutionApproximationKernel, 0, 0);
        std::cout << "[UpdateSolutionApproximationKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[UpdateSolutionApproximationKernel] Set block size: " << "1" << std::endl;
#endif

    UpdateSolutionApproximationKernel <<<1, 1>>> (
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBasisData(),
        fDeviceCorrection,
        fDeviceMaxResidualIndex);
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::UpdateVectorApproximation()
  {
#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, UpdateVectorApproximationKernel, 0, 0);
        std::cout << "[UpdateVectorApproximationKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[UpdateVectorApproximationKernel] Set block size: " << fNLocal << std::endl;
#endif

    UpdateVectorApproximationKernel <<<fNWorkgroups, fNLocal>>> (
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeInfo(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetShapeData(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryInfo(),
        dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundaryData(),
        fDeviceB_iterative,
        fDeviceCorrection,
        fDeviceMaxResidualIndex);

  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::CoalesceData()
  {
    dynamic_cast<KCUDASurfaceContainer&>(fData).ReadBasisData();
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::Finalize()
  {
    dynamic_cast<KCUDASurfaceContainer&>(fData).ReadBasisData();
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::SetResidualVector(const Vector& v)
  {
    fReadResidual = true;

    for( unsigned int i = 0; i<v.Dimension(); i++ ) {
      fHostResidual[i] = v(i);
      fHostB_iterative[i] = fB(i) - fHostResidual[i];
    }

    cudaMemcpy( fDeviceResidual, fHostResidual, dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE), cudaMemcpyHostToDevice );
    cudaMemcpy( fDeviceB_iterative, fHostB_iterative, dynamic_cast<KCUDASurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CU_TYPE), cudaMemcpyHostToDevice );
  }

  template <typename ValueType>
  void KRobinHood_CUDA<ValueType>::GetResidualVector(Vector& v) const
  {
    cudaMemcpy( fHostResidual, fDeviceResidual, fData.GetNBufferedElements()*sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

    for (unsigned int i = 0;i<v.Dimension();i++)
      v[i] = fHostResidual[i];
  }
}

#endif /* KROBINHOOD_CUDA_DEF */
