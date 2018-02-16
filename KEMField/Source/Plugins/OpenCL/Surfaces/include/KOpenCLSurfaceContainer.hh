#ifndef KOPENCLSURFACECONTAINER_DEF
#define KOPENCLSURFACECONTAINER_DEF

#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"
#include "KSurface.hh"
#include "KSortedSurfaceContainer.hh"
#include "KOpenCLInterface.hh"
#include "KOpenCLAction.hh"
#include "KOpenCLData.hh"

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

  class KOpenCLSurfaceContainer :
    public KSortedSurfaceContainer,
    public KOpenCLData
  {
  public:
    KOpenCLSurfaceContainer(const KSurfaceContainer& surfaceContainer);
    virtual ~KOpenCLSurfaceContainer();

    void BuildOpenCLObjects();

    unsigned int GetNBufferedElements() const { return fNBufferedElements; }

    cl::Buffer* GetShapeInfo() const { return fBufferShapeInfo; }
    cl::Buffer* GetShapeData() const { return fBufferShapeData; }
    cl::Buffer* GetBoundaryInfo() const { return fBufferBoundaryInfo; }
    cl::Buffer* GetBoundaryData() const { return fBufferBoundaryData; }
    cl::Buffer* GetBasisData() const { return fBufferBasisData; }

    void ReadBasisData();

    unsigned int GetShapeSize()    const { return fShapeSize; }
    unsigned int GetBoundarySize() const { return fBoundarySize; }
    unsigned int GetBasisSize()    const { return fBasisSize; }

    std::string GetOpenCLFlags() const { return fOpenCLFlags; }

    class FlagGenerator
    {
    public:
      template <class Policy>
      void PerformAction(Type2Type<Policy>)
      {
	std::stringstream s;

	std::string name = Policy::Name();
	for (int pos = 0; name[pos] != '\0'; ++pos)
	  name[pos] = toupper(name[pos]);
	s << " -D " << name << "=";
	fFlag = s.str();
      }

      std::string GetFlag() const { return fFlag; }

    private:
      std::string fFlag;
    };

  protected:
    std::string fOpenCLFlags;

    unsigned int fNBufferedElements;

    unsigned int fShapeSize;
    unsigned int fBoundarySize;
    unsigned int fBasisSize;

    std::vector<cl_short>  fShapeInfo;
    std::vector<CL_TYPE>   fShapeData;
    std::vector<cl_int>    fBoundaryInfo;
    std::vector<CL_TYPE>   fBoundaryData;
    std::vector<CL_TYPE>   fBasisData;

    cl::Buffer* fBufferShapeInfo;
    cl::Buffer* fBufferShapeData;
    cl::Buffer* fBufferBoundaryInfo;
    cl::Buffer* fBufferBoundaryData;
    cl::Buffer* fBufferBasisData;
  };
}

#endif /* KOPENCLSURFACECONTAINER_DEF */
