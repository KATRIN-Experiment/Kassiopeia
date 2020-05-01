#ifndef Kassiopeia_KSReadSet_h_
#define Kassiopeia_KSReadSet_h_

#include "KSReadValue.h"

#include <string>
using std::string;

#include <map>
using std::map;

namespace Kassiopeia
{

template<class XType> class KSReadSet;

template<class XType> class KSReadSet<KSReadValue<XType>>
{
  protected:
    typedef map<std::string, KSReadValue<XType>> ValueMap;
    typedef typename ValueMap::iterator ValueIt;
    typedef typename ValueMap::const_iterator ValueCIt;
    typedef typename ValueMap::value_type ValueEntry;

  public:
    KSReadSet();
    ~KSReadSet();

  public:
    KSReadValue<XType>& Add(const std::string& aLabel);
    KSReadValue<XType>& Get(const std::string& aLabel) const;
    bool Exists(const std::string& aLabel) const;

  protected:
    mutable ValueMap fValueMap;
};

template<class XType> KSReadSet<KSReadValue<XType>>::KSReadSet() : fValueMap() {}
template<class XType> KSReadSet<KSReadValue<XType>>::~KSReadSet() {}

template<class XType> KSReadValue<XType>& KSReadSet<KSReadValue<XType>>::Add(const std::string& aLabel)
{
    auto tIt = fValueMap.find(aLabel);
    if (tIt == fValueMap.end()) {
        return fValueMap[aLabel];
    }
    readermsg(eError) << "value with label <" << aLabel << "> already exists" << eom;
    return tIt->second;
}
template<class XType> KSReadValue<XType>& KSReadSet<KSReadValue<XType>>::Get(const std::string& aLabel) const
{
    auto tIt = fValueMap.find(aLabel);
    if (tIt != fValueMap.end()) {
        return tIt->second;
    }
    readermsg(eError) << "value with label <" << aLabel << "> does not exist" << eom;
    return tIt->second;
}
template<class XType> bool KSReadSet<KSReadValue<XType>>::Exists(const std::string& aLabel) const
{
    auto tIt = fValueMap.find(aLabel);
    if (tIt != fValueMap.end()) {
        return true;
    }
    return false;
}

typedef KSReadSet<KSReadValue<bool>> KSBoolSet;
typedef KSReadSet<KSReadValue<unsigned char>> KSUCharSet;
typedef KSReadSet<KSReadValue<char>> KSCharSet;
typedef KSReadSet<KSReadValue<unsigned short>> KSUShortSet;
typedef KSReadSet<KSReadValue<short>> KSShortSet;
typedef KSReadSet<KSReadValue<unsigned int>> KSUIntSet;
typedef KSReadSet<KSReadValue<int>> KSIntSet;
typedef KSReadSet<KSReadValue<unsigned long>> KSULongSet;
typedef KSReadSet<KSReadValue<long>> KSLongSet;
typedef KSReadSet<KSReadValue<float>> KSFloatSet;
typedef KSReadSet<KSReadValue<double>> KSDoubleSet;
typedef KSReadSet<KSReadValue<KThreeVector>> KSThreeVectorSet;
typedef KSReadSet<KSReadValue<KTwoVector>> KSTwoVectorSet;
typedef KSReadSet<KSReadValue<std::string>> KSStringSet;

}  // namespace Kassiopeia


#endif
