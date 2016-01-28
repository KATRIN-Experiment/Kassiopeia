#ifndef KCUDABUFFERSTREAMER_DEF
#define KCUDABUFFERSTREAMER_DEF

#include "KTypeManipulation.hh"
#include "KTypelist.hh"
#include "KFundamentalTypes.hh"
#include "KFundamentalTypeCounter.hh"

#include "KSurface.hh"

#include "KCUDAInterface.hh"

namespace KEMField
{
/**
* @class KCUDABufferStreamer
*
* @brief A streamer class for inputting surface data into CUDA device memory.
*
* @author Daniel Hilk
*/

class KCUDABufferStreamer;

template <typename Type>
struct KCUDABufferStreamerType
{
    friend inline KCUDABufferStreamer& operator<<(KCUDABufferStreamerType<Type>& d, const Type &x)
    {
        d.AppendToBuffer((CU_TYPE)(x));
        return d.Self();
    }

    friend inline KCUDABufferStreamer& operator>>(KCUDABufferStreamerType<Type>& d, Type &x)
    {
        CU_TYPE tmp;
        d.ExtractFromBuffer(tmp);
        x = tmp;
        return d.Self();
    }

    virtual ~KCUDABufferStreamerType() {}
    virtual void AppendToBuffer(CU_TYPE x) = 0;
    virtual void ExtractFromBuffer(CU_TYPE& x) = 0;
    virtual KCUDABufferStreamer& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes,
        KCUDABufferStreamerType>
KCUDABufferStreamerFundamentalTypes;

class KCUDABufferStreamer : public KCUDABufferStreamerFundamentalTypes
{
public:
    KCUDABufferStreamer() {}
    virtual ~KCUDABufferStreamer() {}
    template <class Streamed>
    void PreStreamInAction(Streamed&) {}
    template <class Streamed>
    void PostStreamInAction(Streamed&) {}
    template <class Streamed>
    void PreStreamOutAction(const Streamed&) {}
    template <class Streamed>
    void PostStreamOutAction(const Streamed&) {}
};

template <class SurfacePolicy>
class KCUDABufferPolicyStreamer : public KCUDABufferStreamer
{
public:
    KCUDABufferPolicyStreamer() : KCUDABufferStreamer(), fWrite(true), fSurfacePolicy(NULL), fBuffer(NULL), fBufferSize(0), fCounter(0) {}
    ~KCUDABufferPolicyStreamer() {}

    template <class Policy>
    void PerformAction(Type2Type<Policy>)
    {
        Policy* policy = static_cast<Policy*>(fSurfacePolicy);
        if (fWrite)
        {
            fTypeCounter.Reset();
            fTypeCounter << *policy;
            *this << *policy;
            for (unsigned int i=fTypeCounter.NumberOfTypes();i<fBufferSize;i++)
                *this << (CU_TYPE)(0.);
        } else {
            fTypeCounter.Reset();
            fTypeCounter << *policy;
            *this >> *policy;
            fCounter += (fBufferSize-fTypeCounter.NumberOfTypes());
        }
    }

    void Reset() { fCounter = 0; }

    void SetToRead() { fWrite = false; }
    void SetToWrite() { fWrite = true; }

    void SetSurfacePolicy(SurfacePolicy* s) { fSurfacePolicy = s; }
    void SetBuffer(CU_TYPE* buffer) { fBuffer = buffer; }
    void SetBufferSize(unsigned int i) { fBufferSize = i; }

protected:
    void AppendToBuffer(CU_TYPE x) { fBuffer[fCounter++] = x; }
    void ExtractFromBuffer(CU_TYPE& x) { x = fBuffer[fCounter++]; }
    virtual KCUDABufferStreamer& Self() { return *this; }

private:
    bool fWrite;
    KFundamentalTypeCounter fTypeCounter;
    SurfacePolicy* fSurfacePolicy;
    CU_TYPE* fBuffer;
    unsigned int fBufferSize;
    unsigned int fCounter;
};
}

#endif /* KCUDABUFFERSTREAMER_DEF */
