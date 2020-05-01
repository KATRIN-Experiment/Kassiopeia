#ifndef KSABUFFER_DEF
#define KSABUFFER_DEF

#include "KFundamentalTypes.hh"
#include "KSAPODConverter.hh"

namespace KEMField
{

/**
* @class KSABuffer
*
* @brief A buffer class for KSA I/O. 
*
* KSABuffer is a buffer for streaming data to and from strings.
*
* @author T.J. Corona
*/

// OK by me -- ES 17/4/13

class KSABuffer;

template<typename Type> struct KSABufferType
{
  public:
    friend inline KSABuffer& operator>>(KSABufferType<Type>& d, Type& x)
    {
        static KSAPODConverter<Type> aPODConverter;
        d.PopEntryFromBuffer();
        aPODConverter.ConvertStringToParameter(d.Entry(), x);
        return d.Self();
    }

    friend inline KSABuffer& operator<<(KSABufferType<Type>& d, const Type& x)
    {
        static KSAPODConverter<const Type> aPODConverter;
        aPODConverter.ConvertParameterToString(d.Entry(), x);
        d.AppendEntryToBuffer();
        return d.Self();
    }

    virtual ~KSABufferType() {}
    virtual const std::string& PopEntryFromBuffer() = 0;
    virtual void AppendEntryToBuffer() = 0;
    virtual std::string& Entry() const = 0;
    virtual KSABuffer& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KSABufferType> KSABufferFundamentalTypes;

class KSABuffer : public KSABufferFundamentalTypes
{
  public:
    KSABuffer() {}
    ~KSABuffer() override {}

    inline std::string& Entry() const override
    {
        return fEntry;
    }
    inline void FillBuffer(std::string s)
    {
        fBufferData = s;
    }
    inline void Clear()
    {
        fBufferData.clear();
    }
    inline void AppendEntryToBuffer() override
    {
        fBufferData.append(fEntry);
        fBufferData.append(&fDataSeparator);
    }
    inline const std::string& PopEntryFromBuffer() override
    {
        size_t pos = fBufferData.find_first_of(fDataSeparator);
        fEntry = fBufferData.substr(0, pos);
        fBufferData = fBufferData.substr(pos + 1, std::string::npos);
        return fEntry;
    }
    inline const std::string& StringifyBuffer() const
    {
        return fBufferData;
    }

    template<class Streamed> void PreStreamInAction(Streamed&)
    {
        size_t pos = fBufferData.find_first_of(fObjectSeparator);
        fBufferData = fBufferData.substr(pos + 1, std::string::npos);
    }
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&)
    {
        fBufferData.append(fObjectSeparator);
    }
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}

  private:
    KSABuffer& Self() override
    {
        return *this;
    }

    std::string fBufferData;
    mutable std::string fEntry;
    static char fDataSeparator;
    static std::string fObjectSeparator;
};
}  // namespace KEMField

#endif /* KSABUFFER_DEF */
