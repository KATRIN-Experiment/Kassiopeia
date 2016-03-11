#include "KCUDAInterface.hh"

#include "KCUDAData.hh"

#include "KEMCout.hh"

#include <stdlib.h>
#include <sstream>


namespace KEMField
{
  KCUDAInterface* KCUDAInterface::fCUDAInterface = 0;

  KCUDAInterface::KCUDAInterface() : fActiveData(NULL)
  {
      InitializeCUDA();
  }

  KCUDAInterface::~KCUDAInterface()
  {

  }

  /**
   * Interface to accessing KCUDAInterface.
   */
  KCUDAInterface* KCUDAInterface::GetInstance()
  {
      if( fCUDAInterface == 0 )
          fCUDAInterface = new KCUDAInterface();

      return fCUDAInterface;
  }

  /**
   * Queries the host for available CUDA devices.
   */
  void KCUDAInterface::InitializeCUDA()
  {
      int devCount( 0 );
      cudaGetDeviceCount( &devCount );
      fDeviceCount = devCount;
      fCUDeviceID = KEMFIELD_DEFAULT_GPU_ID;
  }

  /**
   * Selects a device for use in CUDA calculations.
   */
  void KCUDAInterface::SetGPU( unsigned int i )
  {
      if( i>fDeviceCount ) {
          KEMField::cout << "Cannot set GPU device to ID # "<<i<<", since there are only "<<fDeviceCount<<" devices available." << KEMField::endl;
          return;
      }

      cudaDeviceProp devProp;
      KEMField::cout << "Setting GPU device to ID # "<<i<<" ("<<devProp.name<<") of "<<fDeviceCount<<" available devices." << KEMField::endl;

      fCUDeviceID = i;
  }

  void KCUDAInterface::SetActiveData( KCUDAData* data )
  {
      fActiveData = data;
  }

  KCUDAData* KCUDAInterface::GetActiveData() const
  {
      return fActiveData;
  }
}
