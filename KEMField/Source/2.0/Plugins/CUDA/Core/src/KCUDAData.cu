#include "KCUDAData.hh"

#include <limits>

#include "KCUDAAction.hh"
#include "KEMCout.hh"

namespace KEMField
{
  KCUDAData::KCUDAData():
        fNLocal( std::numeric_limits<unsigned int>::max() ),
        fIsConstructed( false )
  {

  }

  void KCUDAData::ConstructCUDAObjects()
  {
    // Do nothing if this method has already been called
    if (fIsConstructed)
      return;

    // First, construct the kernels that will use the surface buffers (this sets
    // the number of dummy elements in each buffer)
    for (std::vector<const KCUDAAction*>::iterator action = fAssociatedActions.begin();action!=fAssociatedActions.end();++action) {
      if ((*action)->Enabled())
        (*action)->ConstructCUDAKernels();
    }

    if (fNLocal == std::numeric_limits<unsigned int>::max()) {
      std::cout<<"You must first set the number of local streams in a warp before setting the\nbuffers."<<std::endl;
      return;
    }

    // Then, build the actual objects
    BuildCUDAObjects();

    // Finally, we assign the buffers to the actions
    for (std::vector<const KCUDAAction*>::iterator action = fAssociatedActions.begin();action!=fAssociatedActions.end();++action) {
      if((*action)->Enabled())
        (*action)->AssignDeviceMemory();
    }
  }

  void KCUDAData::RegisterAction(const KCUDAAction* action)
  {
    fAssociatedActions.push_back(action);
  }

  void KCUDAData::SetMinimumWorkgroupSizeForKernels(unsigned int nLocal)
  {
    if( nLocal < fNLocal ) {
      if( fIsConstructed )
        KEMField::cout<< "Warning: Buffers have already been constructed with a thread block size "<<fNLocal<<"."<<KEMField::endl;

      fNLocal = nLocal;
    }
  }
}
