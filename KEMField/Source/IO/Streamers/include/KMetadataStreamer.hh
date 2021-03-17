#ifndef KMETADATASTREAMER_DEF
#define KMETADATASTREAMER_DEF

#include "KFundamentalTypes.hh"
#include "KTypeManipulation.hh"

#include <algorithm>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace KEMField
{
class KMetadataStreamer;

template<typename Type> struct KMetadataStreamerType
{
    friend inline KMetadataStreamer& operator<<(KMetadataStreamerType<Type>& d, const Type&)
    {
        d.AddType(FundamentalTypeNames[IndexOf<FundamentalTypes, Type>::value]);
        return d.Self();
    }

    friend inline KMetadataStreamer& operator>>(KMetadataStreamerType<Type>& d, Type&)
    {
        return d.Self();
    }

    virtual ~KMetadataStreamerType() = default;
    virtual void AddType(std::string) = 0;
    virtual KMetadataStreamer& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KMetadataStreamerType> KMetadataStreamerFundamentalTypes;

class KMetadataStreamer : public KMetadataStreamerFundamentalTypes
{
  public:
    KMetadataStreamer() = default;
    ~KMetadataStreamer() override = default;

    static std::string Name()
    {
        return "KMetadataStreamer";
    }

    void open(const std::string& fileName, const std::string& action = "overwrite");
    void close();

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

    std::string StringifyMetadata();

    std::string GetFileSuffix() const
    {
        return ".smd";
    }

    void clear();

  protected:
    KMetadataStreamer& Self() override
    {
        return *this;
    }

    std::fstream fFile;
    bool fIsReading;

    void AddType(std::string type) override;

    template<class Streamed> std::string GetName(Type2Type<Streamed>)
    {
        return Streamed::Name();
    }

    template<class Streamed> std::string GetName(Type2Type<Streamed*>)
    {
        std::stringstream s;
        s << Streamed::Name() << "*(" << fPointerCounts[Streamed::Name()]++ << ")";
        return s.str();
    }

    template<class Streamed> std::string GetName(Type2Type<Streamed&>)
    {
        std::stringstream s;
        s << Streamed::Name() << "&";
        return s.str();
    }

    template<class Streamed> std::string GetName(Type2Type<std::vector<Streamed>>)
    {
        std::stringstream s;
        s << "v[" << GetName(Type2Type<Streamed>()) << "]";
        return s.str();
    }

    template<class Streamed> std::string GetName(Type2Type<std::vector<Streamed>*>)
    {
        std::stringstream s;
        s << "v[" << GetName(Type2Type<Streamed>()) << "]*";
        return s.str();
    }

    template<class Streamed> std::string GetName(Type2Type<std::vector<Streamed>&>)
    {
        std::stringstream s;
        s << "v[" << GetName(Type2Type<Streamed>()) << "]&";
        return s.str();
    }

    using ClassName = std::string;
    using ClassInfo = std::pair<ClassName, unsigned int>;
    using ClassContent = std::vector<ClassInfo>;
    using ClassContentMap = std::map<ClassName, ClassContent>;
    using ClassOrdering = std::list<std::pair<int, ClassContentMap::const_iterator>>;
    using ClassContentList = std::list<ClassContent*>;
    using ClassList = std::list<ClassName>;
    using ClassPointerCounterMap = std::map<ClassName, unsigned int>;

    // We need to deal with the case when a class is revisited, but has a higher
    // ordering value.

    friend bool CompareClassOrdering_weak(ClassOrdering::value_type i, ClassOrdering::value_type j);

    friend bool CompareClassOrdering(ClassOrdering::value_type i, ClassOrdering::value_type j);

    struct KData
    {
        ClassContentMap data;
        ClassList completed;
        ClassOrdering orderedData;
    };

    struct KHierarchy
    {
        ClassList name;
        ClassContentList data;
    };

    KData fData;
    KHierarchy fHierarchy;
    ClassContent fDummy;
    ClassPointerCounterMap fPointerCounts;
};

template<class Streamed> void KMetadataStreamer::PreStreamOutActionForNamed(const Streamed&, Int2Type<true>)
{
    ClassName className = GetName(Type2Type<Streamed>());
    fHierarchy.name.push_front(className);

    AddType(className);

    if (std::find(fData.completed.begin(), fData.completed.end(), className) != fData.completed.end())
        fHierarchy.data.push_front(&fDummy);
    else
        fHierarchy.data.push_front(&(fData.data[className]));
}

template<class Streamed> void KMetadataStreamer::PostStreamOutActionForNamed(const Streamed&, Int2Type<true>)
{
    if (fHierarchy.data.front() == &fDummy)
        fDummy.clear();
    else {
        unsigned int i = fData.orderedData.size();
        fData.orderedData.push_back(std::make_pair(i, fData.data.find(fHierarchy.name.front())));
        fData.completed.push_back(fHierarchy.name.front());
    }
    fHierarchy.name.pop_front();
    fHierarchy.data.pop_front();
}
}  // namespace KEMField

#endif /* KMETADATASTREAMER_DEF */
