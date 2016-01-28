#ifndef KCUDADATA_DEF
#define KCUDADATA_DEF

#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"
#include "KCUDAInterface.hh"

namespace KEMField
{

/**
* @class KCUDAData
*
* @brief A class that holds data to be manipulated by CUDA calls.
*
* @author Daniel Hilk
*/

  class KCUDAAction;

  class KCUDAData
  {
  public:
    KCUDAData();
    virtual ~KCUDAData() {}

    void ConstructCUDAObjects();
    virtual void BuildCUDAObjects() = 0;

    void RegisterAction(const KCUDAAction* action);

    virtual unsigned int GetNBufferedElements() const { return 0; }

    void SetMinimumWorkgroupSizeForKernels(unsigned int nLocal);
    unsigned int GetMinimumWorkgroupSizeForKernels() const { return fNLocal; }

    bool IsConstructed() const { return fIsConstructed; }

  protected:
    std::vector<const KCUDAAction*> fAssociatedActions;
    unsigned int fNLocal;

    bool fIsConstructed;
  };
}

#endif /* KCUDADATA_DEF */
