#ifndef KCUDABOUNDARYINTEGRATOR_DEF
#define KCUDABOUNDARYINTEGRATOR_DEF

#include <iostream>

#include "KCUDABufferStreamer.hh"

#include "KSurface.hh"

#include "KCUDAAction.hh"
#include "KCUDASurfaceContainer.hh"

namespace KEMField
{
  template <class BasisPolicy>
  class KCUDABoundaryIntegrator : public KCUDAAction
  {
  public:
    typedef typename BasisPolicy::ValueType ValueType;

  protected:
    KCUDABoundaryIntegrator( KCUDASurfaceContainer& );
    virtual ~KCUDABoundaryIntegrator();

  protected:
    template <class SourceShape>
    void StreamSourceToDevice(const SourceShape* source) const;

    virtual void ConstructCUDAKernels() const {}
    virtual void AssignDeviceMemory() const;

    mutable short fShapeInfo;
    mutable std::vector<CU_TYPE> fShapeData;
    mutable std::vector<CU_TYPE> fIntegratorData;

    mutable double *fDeviceP;
    mutable short *fDeviceShapeInfo;
    mutable double *fDeviceShapeData;
    mutable double *fDeviceIntegratorData;

    bool fIntegratorDataCopied;

  public:
    CU_TYPE* GetIntegratorData() const { return fDeviceIntegratorData; }
  };

  template <class BasisPolicy>
  KCUDABoundaryIntegrator<BasisPolicy>::KCUDABoundaryIntegrator( KCUDASurfaceContainer& c ) :
    KCUDAAction(c),
    fDeviceP(NULL),
	fDeviceIntegratorData(NULL),
    fDeviceShapeInfo(NULL),
    fDeviceShapeData(NULL)
  {
  }

  template <class BasisPolicy>
  KCUDABoundaryIntegrator<BasisPolicy>::~KCUDABoundaryIntegrator()
  {
    if( fDeviceP ) cudaFree( fDeviceP );
    if( fDeviceIntegratorData ) cudaFree( fDeviceIntegratorData );
    if( fDeviceShapeInfo ) cudaFree( fDeviceShapeInfo );
    if( fDeviceShapeData ) cudaFree( fDeviceShapeData );
  }

  template <class BasisPolicy>
  void KCUDABoundaryIntegrator<BasisPolicy>::AssignDeviceMemory() const
  {
    cudaMalloc( (void**) &fDeviceP, 3*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDeviceShapeInfo, sizeof(short) );
    // Hard-coded arbitrary maximum shape limit
    cudaMalloc( (void**) &fDeviceShapeData, 20*sizeof(CU_TYPE) );
  }

  template <class BasisPolicy>
  template <class SourceShape>
  void KCUDABoundaryIntegrator<BasisPolicy>::StreamSourceToDevice(const SourceShape* source) const
  {
    // Shape Info:
    fShapeInfo = IndexOf<KShapeTypes,SourceShape>::value;
    cudaMemcpy( fDeviceShapeInfo, &fShapeInfo, sizeof(short), cudaMemcpyHostToDevice);

    // Shape Data:

    // First, determine the size of the shape
    static KSurfaceSize<KShape> shapeSize;
    shapeSize.Reset();
    shapeSize.SetSurface(const_cast<SourceShape*>(source));
    shapeSize.PerformAction<SourceShape>(Type2Type<SourceShape>());

    // Then, fill the buffer with the shape information
    static KCUDABufferPolicyStreamer<KShape> shapeStreamer;
    shapeStreamer.Reset();
    shapeStreamer.SetBufferSize(shapeSize.size());
    fShapeData.resize(shapeSize.size(),0.);
    shapeStreamer.SetBuffer(&fShapeData[0]);
    shapeStreamer.SetSurfacePolicy(const_cast<SourceShape*>(source));
    shapeStreamer.PerformAction<SourceShape>(Type2Type<SourceShape>());

    // Finally, send the buffer to the CUDA device
    cudaMemcpy( fDeviceShapeData, &fShapeData[0], shapeSize.size()*sizeof(CU_TYPE), cudaMemcpyHostToDevice );
  }
}

#endif /* KCUDAELECTROSTATICBOUNDARYINTEGRATOR_DEF */
