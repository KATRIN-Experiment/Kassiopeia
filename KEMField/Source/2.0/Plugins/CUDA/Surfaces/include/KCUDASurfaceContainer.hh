#ifndef KCUDASURFACECONTAINER_DEF
#define KCUDASURFACECONTAINER_DEF

#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"
#include "KSurface.hh"
#include "KSortedSurfaceContainer.hh"
#include "KCUDAInterface.hh"
#include "KCUDAAction.hh"
#include "KCUDAData.hh"

namespace KEMField
{

/**
* @class KCUDASurfaceContainer
*
* @brief A data storage class for CUDA.
*
* KCUDASurfaceContainer is a class for collecting surface data into CUDA data.
*
* @author Daniel Hilk
*/

  class KCUDASurfaceContainer :
    public KSortedSurfaceContainer,
    public KCUDAData
  {
  public:
    KCUDASurfaceContainer(const KSurfaceContainer& surfaceContainer);
    virtual ~KCUDASurfaceContainer();

    void BuildCUDAObjects();

    unsigned int GetNBufferedElements() const { return fNBufferedElements; }

    short* GetShapeInfo() const { return fDeviceShapeInfo; }
    CU_TYPE* GetShapeData() const { return fDeviceShapeData; }
    int* GetBoundaryInfo() const { return fDeviceBoundaryInfo; }
    CU_TYPE* GetBoundaryData() const { return fDeviceBoundaryData; }
    CU_TYPE* GetBasisData() const { return fDeviceBasisData; }

    void ReadBasisData();

    unsigned int GetShapeSize()    const { return fShapeSize; }
    unsigned int GetBoundarySize() const { return fBoundarySize; }
    unsigned int GetBasisSize()    const { return fBasisSize; }

    class FlagGenerator
    {
    public:
      template <class Policy>
      void PerformAction(Type2Type<Policy>)
      {
        std::string name = Policy::Name();

        for (int pos = 0; name[pos] != '\0'; ++pos)
          name[pos] = toupper(name[pos]);

        fFlag = name;
       }


      std::string GetFlag() const { return fFlag; }

    private:
      std::string fFlag;
    };

  protected:
    unsigned int fNBufferedElements;

    unsigned int fShapeSize;
    unsigned int fBoundarySize;
    unsigned int fBasisSize;

    std::vector<short>  fShapeInfo;
    std::vector<CU_TYPE>   fShapeData;
    std::vector<int>    fBoundaryInfo;
    std::vector<CU_TYPE>   fBoundaryData;
    std::vector<CU_TYPE>   fBasisData;

    short* fDeviceShapeInfo;
    CU_TYPE* fDeviceShapeData;
    int* fDeviceBoundaryInfo;
    CU_TYPE* fDeviceBoundaryData;
    CU_TYPE* fDeviceBasisData;
  };
}

#endif /* KCUDASURFACECONTAINER_DEF */
