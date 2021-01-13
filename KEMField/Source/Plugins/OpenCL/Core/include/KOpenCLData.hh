#ifndef KOPENCLDATA_DEF
#define KOPENCLDATA_DEF

#include "KFundamentalTypeCounter.hh"
#include "KFundamentalTypes.hh"
#include "KOpenCLInterface.hh"

namespace KEMField
{

/**
* @class KOpenCLData
*
* @brief A class that holds data to be manipulated by OpenCL calls.
*
* @author T.J. Corona
*/

class KOpenCLAction;

class KOpenCLData
{
  public:
    KOpenCLData();
    virtual ~KOpenCLData() = default;

    void ConstructOpenCLObjects();
    virtual void BuildOpenCLObjects() = 0;

    void RegisterAction(const KOpenCLAction* action);

    virtual unsigned int GetNBufferedElements() const
    {
        return 0;
    }

    virtual std::string GetOpenCLFlags() const
    {
        return std::string("");
    }

    void SetMinimumWorkgroupSizeForKernels(unsigned int nLocal);
    unsigned int GetMinimumWorkgroupSizeForKernels() const
    {
        return fNLocal;
    }

    bool IsConstructed() const
    {
        return fIsConstructed;
    }

  protected:
    std::vector<const KOpenCLAction*> fAssociatedActions;
    unsigned int fNLocal;

    bool fIsConstructed;
};
}  // namespace KEMField

#endif /* KOPENCLDATA_DEF */
