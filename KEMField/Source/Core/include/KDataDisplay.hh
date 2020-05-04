#ifndef KDATADISPLAY_DEF
#define KDATADISPLAY_DEF

#include "KFundamentalTypes.hh"
#include "KTypeManipulation.hh"

#include <complex>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

namespace KEMField
{

/**
 * @struct KDataDisplay
 *
 * @brief KEMField's standard output stream.
 *
 * KDataDisplay is functionally identical to std::cout, with the addition of
 * special formatting for KEMField objects.  It can also be used to pipe output
 * to another messaging or logging system.
 *
 * @author T.J. Corona
 */

extern const std::string FundamentalTypeNames[14];

template<class Stream> class KDataDisplay;

template<typename Type, class Stream> struct KDataDisplayType;

struct KNullStream
{};

template<typename Type> struct KDataDisplayType<Type, KNullStream>
{
    friend inline KDataDisplay<KNullStream>& operator<<(KDataDisplayType<Type, KNullStream>& d, const Type)
    {
        return d.Self();
    }

    virtual ~KDataDisplayType() {}
    virtual KDataDisplay<KNullStream>& Self() = 0;
};

template<typename Type, class Stream> struct KDataDisplayType
{
    friend inline KDataDisplay<Stream>& operator<<(KDataDisplayType<Type, Stream>& d, const Type x)
    {
        if (!d.Verbose())
            return d.Self();

        if (d.Level() == 0)
            d.GetStream() << x;
        else {
            for (int i = 0; i < d.Level(); i++)
                d.GetStream() << d.Separator();
            d.GetStream() << "<" << d.AssignmentCounter(d.Level() - 1) << ">("
                          << FundamentalTypeNames[IndexOf<FundamentalTypes, Type>::value] << ")" << x << "<\\"
                          << d.AssignmentCounter(d.Level() - 1) << ">\n";
            d.IncrementAssignmentCounter(d.Level() - 1);
        }
        return d.Self();
    }

    virtual ~KDataDisplayType() {}
    virtual int Level() const = 0;
    virtual Stream& GetStream() = 0;
    virtual std::string& Separator() const = 0;
    virtual unsigned int AssignmentCounter(int) const = 0;
    virtual void IncrementAssignmentCounter(int) = 0;
    virtual bool Verbose() const = 0;
    virtual KDataDisplay<Stream>& Self() = 0;
};

template<class Stream> class KDataDisplayOtherTypes
{
  public:
    KDataDisplayOtherTypes() {}
    virtual ~KDataDisplayOtherTypes() {}

    // function that takes a custom stream, and returns it
    typedef KDataDisplay<Stream>& (*KDataDisplayManipulator)(KDataDisplay<Stream>&);

    // take in a function with the custom signature
    KDataDisplay<Stream>& operator<<(KDataDisplayManipulator manip)
    {
        // call the function, and return it's value
        if (Verbose())
            return manip(Self());
        else
            return Self();
    }

    typedef std::basic_ostream<char, std::char_traits<char>> CoutType;

    // this is the function signature of std::endl
    typedef CoutType& (*StandardEndLine)(CoutType&);

    // define an operator<< to take in std::endl
    KDataDisplay<Stream>& operator<<(StandardEndLine manip)
    {
        // call the function, but we cannot return it's value
        if (Verbose())
            manip(std::cout);

        return Self();
    }

    KDataDisplay<Stream>& operator<<(const char* c)
    {
        if (Verbose())
            GetStream() << c;
        return Self();
    }

    virtual bool Verbose() const = 0;
    virtual Stream& GetStream() = 0;
    virtual KDataDisplay<Stream>& Self() = 0;
};

template<> class KDataDisplayOtherTypes<KNullStream>
{
  public:
    KDataDisplayOtherTypes() {}
    virtual ~KDataDisplayOtherTypes() {}

    // function that takes a custom stream, and returns it
    typedef KDataDisplay<KNullStream>& (*KDataDisplayManipulator)(KDataDisplay<KNullStream>&);

    // take in a function with the custom signature
    KDataDisplay<KNullStream>& operator<<(KDataDisplayManipulator)
    {
        return Self();
    }

    typedef std::basic_ostream<char, std::char_traits<char>> CoutType;

    // this is the function signature of std::endl
    typedef CoutType& (*StandardEndLine)(CoutType&);

    // define an operator<< to take in std::endl
    KDataDisplay<KNullStream>& operator<<(StandardEndLine)
    {
        return Self();
    }

    KDataDisplay<KNullStream>& operator<<(const char*)
    {
        return Self();
    }

    virtual KNullStream& GetStream() = 0;
    virtual KDataDisplay<KNullStream>& Self() = 0;
};

template<class Stream>
class KDataDisplay :
    public KGenScatterHierarchyWithParameter<KEMField::FundamentalTypes, Stream, KDataDisplayType>,
    public KDataDisplayOtherTypes<Stream>
{
  public:
    KDataDisplay();
    ~KDataDisplay() override
    {
        Reset();
    }

    void Verbose(bool isVerbose)
    {
        fVerbose = isVerbose;
    }
    bool Verbose() const override
    {
        return fVerbose;
    }

  protected:
    KDataDisplay<Stream>& Self() override
    {
        return *this;
    }
    void Reset(int i = -1);

    template<class Streamed> std::string GetObjectName(Type2Type<Streamed>)
    {
        return Streamed::Name();
    }
    template<class Streamed> std::string GetObjectName(Type2Type<Streamed*>)
    {
        std::stringstream s;
        s << Streamed::Name() << "*";
        return s.str();
    }
    template<class Streamed> std::string GetObjectName(Type2Type<Streamed&>)
    {
        std::stringstream s;
        s << Streamed::Name() << "&";
        return s.str();
    }
    template<class Streamed> std::string GetObjectName(Type2Type<std::vector<Streamed>>)
    {
        std::stringstream s;
        s << "(std::vector)[" << GetObjectName(Type2Type<Streamed>()) << "]";
        return s.str();
    }
    template<class Streamed> std::string GetObjectName(Type2Type<std::vector<Streamed>*>)
    {
        std::stringstream s;
        s << "(std::vector)[" << GetObjectName(Type2Type<Streamed>()) << "]*";
        return s.str();
    }
    template<class Streamed> std::string GetObjectName(Type2Type<std::vector<Streamed>&>)
    {
        std::stringstream s;
        s << "(std::vector)[" << GetObjectName(Type2Type<Streamed>()) << "]&";
        return s.str();
    }

  public:
    template<class Streamed> void PreStreamInAction(Streamed&) {}
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed& s)
    {
        return PreStreamOutActionForNamed(s, Int2Type<IsNamed<Streamed>::Is>());
    }
    template<class Streamed> void PostStreamOutAction(const Streamed& s)
    {
        return PostStreamOutActionForNamed(s, Int2Type<IsNamed<Streamed>::Is>());
    }

    template<class Streamed> void PreStreamOutActionForNamed(const Streamed&, Int2Type<false>) {}
    template<class Streamed> void PostStreamOutActionForNamed(const Streamed&, Int2Type<false>) {}

    template<class Streamed> void PreStreamOutActionForNamed(const Streamed& s, Int2Type<true>);
    template<class Streamed> void PostStreamOutActionForNamed(const Streamed&, Int2Type<true>);

    void flush()
    {
        fStream.flush();
    }

    Stream& GetStream() override
    {
        return fStream;
    }

  protected:
    Stream fStream;

    int fLevel;
    std::vector<unsigned int> fAssignmentCounter;

    static std::string fSeparator;

    bool fVerbose;

  public:
    int Level() const override
    {
        return fLevel;
    };
    std::string& Separator() const override
    {
        return fSeparator;
    }
    unsigned int AssignmentCounter(int i) const override
    {
        return fAssignmentCounter.at(i);
    }
    void IncrementAssignmentCounter(int i) override
    {
        fAssignmentCounter.at(i)++;
    }


    // special case for std::complex
  public:
    template<typename Type>
    friend KDataDisplay<Stream>& operator<<(KDataDisplay<Stream>& s, const std::complex<Type>& c)
    {
        std::stringstream ss;
        ss << c;
        s << ss.str();
        return s;
    }
};

// define the custom endl for this stream
template<class Stream> static inline KDataDisplay<Stream>& endl(KDataDisplay<Stream>& myDisplay)
{
    if (myDisplay.Verbose())
        myDisplay << std::endl;
    return myDisplay;
}

static inline KDataDisplay<KNullStream>& endl(KDataDisplay<KNullStream>& myDisplay)
{
    return myDisplay;
}

template<class Stream>
template<class Streamed>
void KDataDisplay<Stream>::PreStreamOutActionForNamed(const Streamed&, Int2Type<true>)
{
    if (!fVerbose)
        return;
    for (int i = 0; i < fLevel; i++)
        fStream << fSeparator;
    if (fLevel == 0)
        fStream << "\n<" << GetObjectName(Type2Type<Streamed>()) << ">\n";
    else
        fStream << "<" << fAssignmentCounter.at(fLevel - 1) << ">" << GetObjectName(Type2Type<Streamed>()) << "\n";
    fLevel++;
    fAssignmentCounter.push_back(0);
}

template<class Stream>
template<class Streamed>
void KDataDisplay<Stream>::PostStreamOutActionForNamed(const Streamed&, Int2Type<true>)
{
    if (!fVerbose)
        return;
    fLevel--;
    Reset(fLevel);
    for (int i = 0; i < fLevel; i++)
        fStream << fSeparator;

    if (fLevel == 0)
        fStream << "<\\" << GetObjectName(Type2Type<Streamed>()) << ">";
    else
        fStream << "<\\" << fAssignmentCounter.at(fLevel - 1)++ << ">\n";
}

template<class Stream> std::string KDataDisplay<Stream>::fSeparator = "    ";

template<class Stream>
KDataDisplay<Stream>::KDataDisplay() :
    KDataDisplayOtherTypes<Stream>(),
    fStream(std::cout.rdbuf()),
    fLevel(0),
    fVerbose(true)
{
    Reset();
}

template<class Stream> void KDataDisplay<Stream>::Reset(int i)
{
    if (i == -1)
        fAssignmentCounter.clear();
    else
        fAssignmentCounter.at(i) = 0;
}

template<>
class KDataDisplay<KNullStream> :
    public KGenScatterHierarchyWithParameter<KEMField::FundamentalTypes, KNullStream, KDataDisplayType>,
    public KDataDisplayOtherTypes<KNullStream>
{
  public:
    KDataDisplay() {}
    ~KDataDisplay() override {}

  protected:
    KDataDisplay<KNullStream>& Self() override
    {
        return *this;
    }

  public:
    template<class Streamed> void PreStreamInAction(Streamed&) {}
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&) {}
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}

    void flush() {}

    KNullStream& GetStream() override
    {
        return fStream;
    }

  protected:
    KNullStream fStream;
};

}  // namespace KEMField

#endif /* KDATADISPLAY_DEF */
