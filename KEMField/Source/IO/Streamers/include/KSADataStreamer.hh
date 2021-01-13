#ifndef KSADATASTREAMER_DEF
#define KSADATASTREAMER_DEF

#include "KSABuffer.hh"
#include "KSAFileReader.hh"
#include "KSAFileWriter.hh"

#include <algorithm>

namespace KEMField
{

/**
 * @class KSADataStreamer
 *
 * @brief A streamer class for KSA I/O. 
 *
 * KSADataStreamer is a buffer for streaming data in the form of strings.
 *
 * @author T.J. Corona
 */

class KSADataStreamer;

template<typename Type> struct KSADataStreamerType
{
    friend inline KSADataStreamer& operator>>(KSADataStreamerType<Type>& d, Type& x)
    {
        d.Buffer() >> x;
        return d.Self();
    }

    friend inline KSADataStreamer& operator<<(KSADataStreamerType<Type>& d, const Type& x)
    {
        d.Buffer() << x;
        return d.Self();
    }

    virtual ~KSADataStreamerType() = default;
    virtual KSABuffer& Buffer() = 0;
    virtual KSADataStreamer& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KSADataStreamerType> KSADataStreamerFundamentalTypes;

class KSADataStreamer : public KSADataStreamerFundamentalTypes
{
  public:
    KSADataStreamer() : fFlushSize(CHUNK) {}
    ~KSADataStreamer() override = default;

    void open(const std::string& fileName, const std::string& action);
    void close();

    void flush();

    template<class Streamed> void PreStreamInAction(Streamed& s);
    template<class Streamed> void PostStreamInAction(Streamed& s);
    template<class Streamed> void PreStreamOutAction(const Streamed& s)
    {
        fBuffer.PreStreamOutAction(s);
    }
    template<class Streamed> void PostStreamOutAction(const Streamed& s)
    {
        fBuffer.PostStreamOutAction(s);
    }

    std::string GetFileSuffix() const
    {
        return ".zksa";
    }

    KSABuffer& Buffer() override
    {
        return fBuffer;
    }

  protected:
    KSADataStreamer& Self() override
    {
        return *this;
    }

    KSABuffer fBuffer;

    KSAFileReader fReader;
    KSAFileWriter fWriter;

    bool fIsReading;
    unsigned int fFlushSize;
};

template<class Streamed> void KSADataStreamer::PreStreamInAction(Streamed& s)
{
    if (fIsReading) {
        std::string s;
        fReader.GetLine(s);
        fBuffer.FillBuffer(s);
    }
    fBuffer.PreStreamInAction(s);
}

template<class Streamed> void KSADataStreamer::PostStreamInAction(Streamed& s)
{
    if (!fIsReading) {
        if (fBuffer.StringifyBuffer().size() > fFlushSize)
            flush();
    }
    fBuffer.PostStreamInAction(s);
}
}  // namespace KEMField

#endif /* KSADATASTREAMER_DEF */
