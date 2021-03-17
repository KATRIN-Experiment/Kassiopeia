#ifndef KOPENCLSURFACECONTAINER_DEF
#define KOPENCLSURFACECONTAINER_DEF

#include "KFundamentalTypeCounter.hh"
#include "KFundamentalTypes.hh"
#include "KOpenCLAction.hh"
#include "KOpenCLData.hh"
#include "KOpenCLInterface.hh"
#include "KSortedSurfaceContainer.hh"
#include "KSurface.hh"

namespace KEMField
{

/**
* @class KOpenCLSurfaceContainer
*
* @brief A data storage class for OpenCL. 
*
* KOpenCLSurfaceContainer is a class for collecting surface data into OpenCL data.
*
* @author T.J. Corona
*/

class KOpenCLSurfaceContainer : public KSortedSurfaceContainer, public KOpenCLData
{
  public:
    KOpenCLSurfaceContainer(const KSurfaceContainer& surfaceContainer);
    ~KOpenCLSurfaceContainer() override;

    void BuildOpenCLObjects() override;

    unsigned int GetNBufferedElements() const override
    {
        return fNBufferedElements;
    }

    cl::Buffer* GetShapeInfo() const
    {
        return fBufferShapeInfo;
    }
    cl::Buffer* GetShapeData() const
    {
        return fBufferShapeData;
    }
    cl::Buffer* GetBoundaryInfo() const
    {
        return fBufferBoundaryInfo;
    }
    cl::Buffer* GetBoundaryData() const
    {
        return fBufferBoundaryData;
    }
    cl::Buffer* GetBasisData() const
    {
        return fBufferBasisData;
    }

    void ReadBasisData();

    unsigned int GetShapeSize() const
    {
        return fShapeSize;
    }
    unsigned int GetBoundarySize() const
    {
        return fBoundarySize;
    }
    unsigned int GetBasisSize() const
    {
        return fBasisSize;
    }

    std::string GetOpenCLFlags() const override
    {
        return fOpenCLFlags;
    }

    class FlagGenerator
    {
      public:
        template<class Policy> void PerformAction(Type2Type<Policy>)
        {
            std::stringstream s;

            std::string name = Policy::Name();
            for (int pos = 0; name[pos] != '\0'; ++pos)
                name[pos] = toupper(name[pos]);
            s << " -D " << name << "=";
            fFlag = s.str();
        }

        std::string GetFlag() const
        {
            return fFlag;
        }

      private:
        std::string fFlag;
    };

  protected:
    std::string fOpenCLFlags;

    unsigned int fNBufferedElements;

    unsigned int fShapeSize;
    unsigned int fBoundarySize;
    unsigned int fBasisSize;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::vector<cl_short> fShapeInfo;
    std::vector<CL_TYPE> fShapeData;
    std::vector<cl_int> fBoundaryInfo;
    std::vector<CL_TYPE> fBoundaryData;
    std::vector<CL_TYPE> fBasisData;
#pragma GCC diagnostic pop

    cl::Buffer* fBufferShapeInfo;
    cl::Buffer* fBufferShapeData;
    cl::Buffer* fBufferBoundaryInfo;
    cl::Buffer* fBufferBoundaryData;
    cl::Buffer* fBufferBasisData;
};
}  // namespace KEMField

#endif /* KOPENCLSURFACECONTAINER_DEF */
