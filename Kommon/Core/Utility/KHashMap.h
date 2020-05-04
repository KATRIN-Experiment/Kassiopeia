/**
 * @file
 * Contains katrin::KValueCache
 * @date Created on: 06.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KHASHMAP_H_
#define KHASHMAP_H_

#include "KException.h"
#include "KHash.h"
#include "KLogger.h"
#include "KTypeTraits.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>

namespace katrin
{

/**
 * Stores key-value pairs in an unordered hash map.
 * In a hash map the lookup time is constant O(1), meaning not depending on the size of the container.
 * @tparam Input The type of the key.
 * This type must be hashable. Primitive types and std::tuples of those are
 * already supported.
 * @tparam Result The type of the value. Any copy-constructable type will do.
 * @see http://www.boost.org/doc/libs/1_51_0/doc/html/boost/unordered_map.html
 */
template<class Input, class Result, class Hash = std::hash<Input>> class KHashMap
{
  public:
    KHashMap(size_t maxCacheSize = 8192, double maxLoadFactor = 2.0);
    virtual ~KHashMap() {}

    typedef Input CacheKey;

    /**
     * Define the maximum size of the container.
     * After maxSize is reached, the underlying map is purged.
     * @param maxSize
     */
    void SetMaxSize(size_t maxSize);
    size_t Size() const;

    /**
     * Set the maximum load factor (values per hash index within the map).
     * @param factor
     */
    void SetMaxLoadFactor(double factor);
    double LoadFactor() const;

    /**
     * Purge the underlying map.
     */
    void Clear();

    /**
     * Reduce randomly picked elements until only newSize elements are left.
     * @param newSize
     */
    void Reduce(size_t newSize);

    /**
     * Store a new value.
     * @param key
     * @param result
     * @return A reference to the new value.
     */
    const Result& Store(const Input& key, const Result& result);

    /**
     * Check if a value exists in the map.
     * @param key
     * @return
     */
    bool Has(const Input& key) const;

    /**
     * Try to retrieve a stored value (exception safe).
     * @param[in] input The key.
     * @param[out] result A reference where to store the value.
     * @return True if the value existed.
     */
    bool Get(const Input& input, Result& result) const;

    /**
     * Retrieve a stored value;
     * @param key
     * @throw KException If the requested key doesn't exist.
     * @return
     */
    const Result& Get(const Input& key) const;

    /**
     * Get the current efficiency (number of read over number of write accesses).
     * @return
     */
    double Efficiency() const;

    /**
     * Get the current memory consumption in bytes.
     * @return
     */
    uint64_t Memory() const;

    /**
     * Allow the ValueCache to print with given interval a status about it's efficiency,
     * load factor and memory consumption.
     * @param reportFreq
     * @param cacheName
     */
    void EnableReporting(uint32_t reportFreq = 1000, const std::string& cacheName = "");

  private:
    typedef std::unordered_map<Input, Result, Hash> CacheMap;
    typedef typename CacheMap::iterator CacheMapIterator;
    typedef typename CacheMap::const_iterator CacheMapConstIterator;

    CacheMap fHashMap;
    size_t fMaxCacheSize;

    mutable uint64_t fNRead;
    mutable uint64_t fNWrite;

    std::string fCacheName;
    uint32_t fReportingFrequency;
};

template<class Input, class Result, class Hash>
inline KHashMap<Input, Result, Hash>::KHashMap(size_t maxCacheSize, double maxLoadFactor) :
    fHashMap(),
    fMaxCacheSize(maxCacheSize),
    fNRead(0),
    fNWrite(0),
    fReportingFrequency(0)
{
    fHashMap.max_load_factor(maxLoadFactor);
    //    fCache->rehash(fMaxCacheSize);
}

template<class Input, class Result, class Hash> inline size_t KHashMap<Input, Result, Hash>::Size() const
{
    return fHashMap.size();
}

template<class Input, class Result, class Hash> inline double KHashMap<Input, Result, Hash>::LoadFactor() const
{
    return fHashMap.load_factor();
}

template<class Input, class Result, class Hash>
inline void KHashMap<Input, Result, Hash>::SetMaxLoadFactor(double factor)
{
    fHashMap.max_load_factor(factor);
}

template<class Input, class Result, class Hash> inline void KHashMap<Input, Result, Hash>::SetMaxSize(size_t maxSize)
{
    fMaxCacheSize = maxSize;
    if (fHashMap.size() >= fMaxCacheSize)
        Clear();
}

template<class Input, class Result, class Hash> inline void KHashMap<Input, Result, Hash>::Clear()
{
    fHashMap.clear();
    fNRead = 0;
    fNWrite = 0;
}

template<class Input, class Result, class Hash> inline void KHashMap<Input, Result, Hash>::Reduce(size_t newSize)
{
    if (newSize == 0) {
        fHashMap.clear();
        return;
    }

    auto it = fHashMap.begin();
    while (fHashMap.size() > newSize && it != fHashMap.end()) {
        it = fHashMap.erase(it);
    }
}

template<class Input, class Result, class Hash>
inline const Result& KHashMap<Input, Result, Hash>::Store(const Input& input, const Result& result)
{
    ++fNWrite;
    if (fHashMap.size() >= fMaxCacheSize)
        Reduce(fMaxCacheSize / 2);

    std::pair<CacheMapIterator, bool> insertion = fHashMap.insert(std::make_pair(input, result));
    if (!insertion.second)
        insertion.first->second = result;

    KLOGGER(logger, "common.valuecache");
    if (fReportingFrequency > 0 && fNWrite % fReportingFrequency == 0)
        KDEBUG(logger,
               "VCache [" << fCacheName << "] Size: " << Size() << ", Efficiency: " << Efficiency()
                          << ", Memory: " << Memory() << ", Avg. Load Factor: " << fHashMap.load_factor()
                          << ", Max. Load Factor: " << fHashMap.max_load_factor());

    return insertion.first->second;
}

template<class Input, class Result, class Hash> inline bool KHashMap<Input, Result, Hash>::Has(const Input& input) const
{
    return fHashMap.find(input) != fHashMap.end();
}

template<class Input, class Result, class Hash>
inline bool KHashMap<Input, Result, Hash>::Get(const Input& input, Result& result) const
{
    auto it = fHashMap.find(input);

    ++fNRead;

    //    KLOGGER(logger, "common.valuecache");
    //    if (fReportingFrequency > 0 && fNRead % fReportingFrequency == 0) {
    //        KDEBUG(logger, "VCache [" << fCacheName << "] Size: " << Size() << ", Efficiency: " << Efficiency() << ", Memory: " << Memory()
    //            << ", Avg. Load Factor: " << fHashMap.load_factor() << ", Max. Load Factor: " << fHashMap.max_load_factor() );
    //    }

    if (it != fHashMap.end()) {
        result = it->second;
        return true;
    }
    else {
        return false;
    }
}

template<class Input, class Result, class Hash>
inline const Result& KHashMap<Input, Result, Hash>::Get(const Input& input) const
{
    auto it = fHashMap.find(input);

    if (it == fHashMap.end())
        throw KException() << "Unknown hash map key.";

    return it->second;
}

template<class Input, class Result, class Hash> inline double KHashMap<Input, Result, Hash>::Efficiency() const
{
    return (fNWrite == 0) ? 0.0 : (double) fNRead / (double) fNWrite;
}

template<class Input, class Result, class Hash> inline uint64_t KHashMap<Input, Result, Hash>::Memory() const
{
    return sizeof(typename CacheMap::value_type) * fHashMap.size();
}

template<class Input, class Result, class Hash>
void KHashMap<Input, Result, Hash>::EnableReporting(uint32_t reportFreq, const std::string& cacheName)
{
    fReportingFrequency = reportFreq;
    fCacheName = cacheName;
}

} /* namespace katrin */
#endif /* KHASHMAP_H_ */
