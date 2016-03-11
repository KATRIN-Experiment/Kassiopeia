#ifndef KCUDAINTERFACE_DEF
#define KCUDAINTERFACE_DEF

#include <cuda.h>
#include <cuda_runtime.h>

#include "kEMField_cuda_defines.h"

#include <string>
#include <vector>


namespace KEMField{

  class KCUDAData;

  class KCUDAInterface
  {
  public:
    static KCUDAInterface* GetInstance();

    unsigned int GetNumberOfDevices() const { return fDeviceCount; };

    void SetGPU( unsigned int i );

    void SetActiveData( KCUDAData* data );
    KCUDAData* GetActiveData() const;

  protected:
    KCUDAInterface();
    virtual ~KCUDAInterface();

    void InitializeCUDA();

    static KCUDAInterface* fCUDAInterface;

    unsigned int        fDeviceCount;
    unsigned int        fCUDeviceID;
    mutable KCUDAData   *fActiveData;
  };

}

#endif /* KCUDAINTERFACE_DEF */
