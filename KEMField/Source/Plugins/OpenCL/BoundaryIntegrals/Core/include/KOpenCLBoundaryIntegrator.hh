#ifndef KOPENCLBOUNDARYINTEGRATOR_DEF
#define KOPENCLBOUNDARYINTEGRATOR_DEF

#include "KOpenCLAction.hh"
#include "KOpenCLBufferStreamer.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KSurface.hh"

#include <iostream>

namespace KEMField
{
template<class BasisPolicy> class KOpenCLBoundaryIntegrator : public KOpenCLAction
{
  public:
    typedef typename BasisPolicy::ValueType ValueType;

  protected:
    KOpenCLBoundaryIntegrator(KOpenCLSurfaceContainer&);
    virtual ~KOpenCLBoundaryIntegrator();

  public:
    std::string GetOpenCLFlags() const
    {
        return fOpenCLFlags;
    }

    virtual std::string OpenCLFile() const = 0;

  protected:
    template<class SourceShape> void StreamSourceToBuffer(const SourceShape* source) const;

    virtual void ConstructOpenCLKernels() const {}
    virtual void AssignBuffers() const;

    mutable cl_short fShapeInfo;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    mutable std::vector<CL_TYPE> fShapeData;
#pragma GCC diagnostic pop
    mutable std::string fOpenCLFlags;

    mutable cl::Buffer* fBufferP;
    mutable cl::Buffer* fBufferShapeInfo;
    mutable cl::Buffer* fBufferShapeData;
};

template<class BasisPolicy>
KOpenCLBoundaryIntegrator<BasisPolicy>::KOpenCLBoundaryIntegrator(KOpenCLSurfaceContainer& c) :
    KOpenCLAction(c),
    fBufferP(NULL),
    fBufferShapeInfo(NULL),
    fBufferShapeData(NULL)
{
    std::stringstream options;
    options << "-I " << KOpenCLInterface::GetInstance()->GetKernelPath() << "/";
    fOpenCLFlags = options.str();
}

template<class BasisPolicy> KOpenCLBoundaryIntegrator<BasisPolicy>::~KOpenCLBoundaryIntegrator()
{
    if (fBufferP)
        delete fBufferP;
    if (fBufferShapeInfo)
        delete fBufferShapeInfo;
    if (fBufferShapeData)
        delete fBufferShapeData;
}

template<class BasisPolicy> void KOpenCLBoundaryIntegrator<BasisPolicy>::AssignBuffers() const
{
    fBufferP = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, 3 * sizeof(CL_TYPE));
    fBufferShapeInfo =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sizeof(cl_short));
    fBufferShapeData = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_READ_ONLY,
                                      // Hard-coded arbitrary maximum shape limit
                                      20 * sizeof(CL_TYPE));
}

template<class BasisPolicy>
template<class SourceShape>
void KOpenCLBoundaryIntegrator<BasisPolicy>::StreamSourceToBuffer(const SourceShape* source) const
{
    // Shape Info:
    fShapeInfo = IndexOf<KShapeTypes, SourceShape>::value;

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferShapeInfo,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sizeof(cl_short),
                                                                   &fShapeInfo);

    // Shape Data:

    // First, determine the size of the shape
    static KSurfaceSize<KShape> shapeSize;
    shapeSize.Reset();
    shapeSize.SetSurface(const_cast<SourceShape*>(source));
    shapeSize.PerformAction<SourceShape>(Type2Type<SourceShape>());

    // Then, fill the buffer with the shape information
    static KOpenCLBufferPolicyStreamer<KShape> shapeStreamer;
    shapeStreamer.Reset();
    shapeStreamer.SetBufferSize(shapeSize.size());
    fShapeData.resize(shapeSize.size(), 0.);
    shapeStreamer.SetBuffer(&fShapeData[0]);
    shapeStreamer.SetSurfacePolicy(const_cast<SourceShape*>(source));
    shapeStreamer.PerformAction<SourceShape>(Type2Type<SourceShape>());

    // Finally, send the buffer to the compute device
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferShapeData,
                                                                   CL_TRUE,
                                                                   0,
                                                                   shapeSize.size() * sizeof(CL_TYPE),
                                                                   &fShapeData[0]);
}
}  // namespace KEMField

#endif /* KOPENCLELECTROSTATICBOUNDARYINTEGRATOR_DEF */
