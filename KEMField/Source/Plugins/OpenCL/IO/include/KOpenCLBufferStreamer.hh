#ifndef KOPENCLBUFFERSTREAMER_DEF
#define KOPENCLBUFFERSTREAMER_DEF

#include "KFundamentalTypeCounter.hh"
#include "KFundamentalTypes.hh"
#include "KOpenCLInterface.hh"
#include "KSurface.hh"
#include "KTypeManipulation.hh"
#include "KTypelist.hh"

namespace KEMField
{
/**
* @class KOpenCLBufferStreamer
*
* @brief A streamer class for inputting surface data into OpenCL buffers. 
*
* @author T.J. Corona
*/

class KOpenCLBufferStreamer;

template<typename Type> struct KOpenCLBufferStreamerType
{
    friend inline KOpenCLBufferStreamer& operator<<(KOpenCLBufferStreamerType<Type>& d, const Type& x)
    {
        d.AppendToBuffer((CL_TYPE)(x));
        return d.Self();
    }

    friend inline KOpenCLBufferStreamer& operator>>(KOpenCLBufferStreamerType<Type>& d, Type& x)
    {
        CL_TYPE tmp;
        d.ExtractFromBuffer(tmp);
        x = tmp;
        return d.Self();
    }

    virtual ~KOpenCLBufferStreamerType() {}
    virtual void AppendToBuffer(CL_TYPE x) = 0;
    virtual void ExtractFromBuffer(CL_TYPE& x) = 0;
    virtual KOpenCLBufferStreamer& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KOpenCLBufferStreamerType>
    KOpenCLBufferStreamerFundamentalTypes;

class KOpenCLBufferStreamer : public KOpenCLBufferStreamerFundamentalTypes
{
  public:
    KOpenCLBufferStreamer() {}
    virtual ~KOpenCLBufferStreamer() {}
    template<class Streamed> void PreStreamInAction(Streamed&) {}
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&) {}
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}
};

template<class SurfacePolicy> class KOpenCLBufferPolicyStreamer : public KOpenCLBufferStreamer
{
  public:
    KOpenCLBufferPolicyStreamer() :
        KOpenCLBufferStreamer(),
        fWrite(true),
        fSurfacePolicy(NULL),
        fBuffer(NULL),
        fBufferSize(0),
        fCounter(0)
    {}
    ~KOpenCLBufferPolicyStreamer() {}

    template<class Policy> void PerformAction(Type2Type<Policy>)
    {
        Policy* policy = static_cast<Policy*>(fSurfacePolicy);
        if (fWrite) {
            fTypeCounter.Reset();
            fTypeCounter << *policy;
            *this << *policy;
            for (unsigned int i = fTypeCounter.NumberOfTypes(); i < fBufferSize; i++)
                *this << (CL_TYPE)(0.);
        }
        else {
            fTypeCounter.Reset();
            fTypeCounter << *policy;
            *this >> *policy;
            fCounter += (fBufferSize - fTypeCounter.NumberOfTypes());
        }
    }

    void Reset()
    {
        fCounter = 0;
    }

    void SetToRead()
    {
        fWrite = false;
    }
    void SetToWrite()
    {
        fWrite = true;
    }

    void SetSurfacePolicy(SurfacePolicy* s)
    {
        fSurfacePolicy = s;
    }
    void SetBuffer(CL_TYPE* buffer)
    {
        fBuffer = buffer;
    }
    void SetBufferSize(unsigned int i)
    {
        fBufferSize = i;
    }

  protected:
    void AppendToBuffer(CL_TYPE x)
    {
        fBuffer[fCounter++] = x;
    }
    void ExtractFromBuffer(CL_TYPE& x)
    {
        x = fBuffer[fCounter++];
    }
    virtual KOpenCLBufferStreamer& Self()
    {
        return *this;
    }

  private:
    bool fWrite;
    KFundamentalTypeCounter fTypeCounter;
    SurfacePolicy* fSurfacePolicy;
    CL_TYPE* fBuffer;
    unsigned int fBufferSize;
    unsigned int fCounter;
};
}  // namespace KEMField

#endif /* KOPENCLBUFFERSTREAMER_DEF */
