#ifndef Kassiopeia_KSReadSet_h_
#define Kassiopeia_KSReadSet_h_

#include "KSReadValue.h"

#include <map>
#include <string>

namespace Kassiopeia
{

template<class XType> class KSReadSet;

template<class XType> class KSReadSet<KSReadValue<XType>>
{
  protected:
    using ValueMap = std::map<std::string, KSReadValue<XType>>;
    using ValueIt = typename ValueMap::iterator;
    using ValueCIt = typename ValueMap::const_iterator;
    using ValueEntry = typename ValueMap::value_type;

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
template<class XType> KSReadSet<KSReadValue<XType>>::~KSReadSet() = default;

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

using KSBoolSet = KSReadSet<KSReadValue<bool>>;
using KSUCharSet = KSReadSet<KSReadValue<unsigned char>>;
using KSCharSet = KSReadSet<KSReadValue<char>>;
using KSUShortSet = KSReadSet<KSReadValue<unsigned short>>;
using KSShortSet = KSReadSet<KSReadValue<short>>;
using KSUIntSet = KSReadSet<KSReadValue<unsigned int>>;
using KSIntSet = KSReadSet<KSReadValue<int>>;
using KSULongSet = KSReadSet<KSReadValue<unsigned long>>;
using KSLongSet = KSReadSet<KSReadValue<long>>;
using KSLongLongSet = KSReadSet<KSReadValue<long long>>;
using KSFloatSet = KSReadSet<KSReadValue<float>>;
using KSDoubleSet = KSReadSet<KSReadValue<double>>;
using KSThreeVectorSet = KSReadSet<KSReadValue<katrin::KThreeVector>>;
using KSTwoVectorSet = KSReadSet<KSReadValue<katrin::KTwoVector>>;
using KSThreeMatrixSet = KSReadSet<KSReadValue<katrin::KThreeMatrix>>;
using KSTwoMatrixSet = KSReadSet<KSReadValue<katrin::KTwoMatrix>>;
using KSStringSet = KSReadSet<KSReadValue<std::string>>;

}  // namespace Kassiopeia


#endif
