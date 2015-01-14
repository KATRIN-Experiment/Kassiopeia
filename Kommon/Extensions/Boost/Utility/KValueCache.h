/**
 * @file
 * Contains katrin::KValueCache
 * @date Created on: 06.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KVALUECACHE_H_
#define KVALUECACHE_H_

#include "KLogger.h"

#include "KException.h"

#include <cstdlib>
#include <algorithm>
#include <string>

#include <boost/version.hpp>
#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/functional/hash.hpp>
#include <boost/utility.hpp>
#include <boost/array.hpp>
#include <boost/version.hpp>

namespace boost
{
    namespace tuples
    {
        namespace detail
        {

            template< class Tuple, size_t Index = length< Tuple >::value - 1 >
            struct HashValueImpl
            {
                    static inline void apply( size_t& seed, Tuple const& tuple )
                    {
                        HashValueImpl< Tuple, Index - 1 >::apply( seed, tuple );
                        boost::hash_combine( seed, tuple.template get< Index >() );
                    }
            };

            template< class Tuple >
            struct HashValueImpl< Tuple, 0 >
            {
                    static inline void apply( size_t& seed, Tuple const& tuple )
                    {
                        boost::hash_combine( seed, tuple.template get< 0 >() );
                    }
            };
        }

        template< class Tuple >
        inline size_t hash_value( Tuple const& tuple )
        {
            size_t seed = 0;
            detail::HashValueImpl< Tuple >::apply( seed, tuple );
            return seed;
        }

    }

#if BOOST_VERSION < 105000

template <typename ValueT, size_t D>
inline size_t hash_value(const boost::array<ValueT, D>& arr)
{
    return boost::hash_range(arr.begin(), arr.end());
}

#endif

}

namespace katrin
{

/**
 * Stores key-value pairs in an unordered hash map.
 * In a hash map the lookup time is constant O(1), meaning not depending on the size of the container.
 * @tparam Input The type of the key.
 * This type must be hashable by the boost::hash library. Primitive types and boost::tuples of those are
 * already supported.
 * @tparam Result The type of the value. Any copy-constructable type will do.
 * @see http://www.boost.org/doc/libs/1_51_0/doc/html/boost/unordered_map.html
 */
template< class Input, class Result >
class KValueCache
{
    public:
        KValueCache( std::size_t maxCacheSize = 8192, double maxLoadFactor = 2.0 );
        virtual ~KValueCache() { }

        typedef Input CacheKey;

        /**
         * Define the maximum size of the container.
         * After maxSize is reached, the underlying map is purged.
         * @param maxSize
         */
        void SetMaxSize( std::size_t maxSize );
        std::size_t Size() const;

        /**
         * Set the maximum load factor (values per hash index within the map).
         * @param factor
         */
        void SetMaxLoadFactor( double factor );
        double LoadFactor() const;

        /**
         * Purge the underlying map.
         */
        void Clear();

        /**
         * Reduce randomly picked elements until only newSize elements are left.
         * @param newSize
         */
        void Reduce( std::size_t newSize );

        /**
         * Store a new value.
         * @param key
         * @param result
         * @return A reference to the new value.
         */
        const Result& Store( const Input& key, const Result& result );

        /**
         * Check if a value exists in the map.
         * @param key
         * @return
         */
        bool Find( const Input& key ) const;

        /**
         * Try to retrieve a stored value (exception safe).
         * @param[in] input The key.
         * @param[out] result A reference where to store the value.
         * @return True if the value existed.
         */
        bool Get( const Input& input, Result& result ) const;

        /**
         * Retrieve a stored value;
         * @param key
         * @throw KException If the requested key doesn't exist.
         * @return
         */
        const Result& Get( const Input& key ) const throw (KException);

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
        void EnableReporting( uint32_t reportFreq = 1000, const std::string& cacheName = "" );

    private:
        typedef boost::unordered_map< Input, Result > CacheMap;
        typedef typename boost::unordered_map< Input, Result >::iterator CacheMapIterator;
        typedef typename boost::unordered_map< Input, Result >::const_iterator CacheMapConstIterator;

        CacheMap fHashMap;
        std::size_t fMaxCacheSize;

        mutable uint64_t fNRead;
        mutable uint64_t fNWrite;

        std::string fCacheName;
        uint32_t fReportingFrequency;
};

template< class Input, class Result >
inline KValueCache< Input, Result >::KValueCache( std::size_t maxCacheSize, double maxLoadFactor ) :
        fHashMap(),
        fMaxCacheSize( maxCacheSize ),
        fNRead( 0 ),
        fNWrite( 0 ),
        fReportingFrequency( 0 )
{
    fHashMap.max_load_factor( maxLoadFactor );
//    fCache->rehash(fMaxCacheSize);
}

template< class Input, class Result >
inline std::size_t KValueCache< Input, Result >::Size() const
{
    return fHashMap.size();
}

template< class Input, class Result >
inline double KValueCache< Input, Result >::LoadFactor() const
{
    return fHashMap.load_factor();
}

template< class Input, class Result >
inline void KValueCache< Input, Result >::SetMaxLoadFactor( double factor )
{
    fHashMap.max_load_factor( factor );
}

template< class Input, class Result >
inline void KValueCache< Input, Result >::SetMaxSize( std::size_t maxSize )
{
    fMaxCacheSize = maxSize;
    if( fHashMap.size() >= fMaxCacheSize )
        Clear();
}

template< class Input, class Result >
inline void KValueCache< Input, Result >::Clear()
{
    fHashMap.clear();
    fNRead = 0;
    fNWrite = 0;
}

template< class Input, class Result >
inline void KValueCache< Input, Result >::Reduce(std::size_t newSize)
{
    if (newSize == 0) {
        fHashMap.clear();
        return;
    }

    CacheMapIterator it = fHashMap.begin();
    while (fHashMap.size() > newSize && it != fHashMap.end()) {
        it = fHashMap.erase(it);
    }
}

template< class Input, class Result >
inline const Result& KValueCache< Input, Result >::Store( const Input& input, const Result& result )
{
    ++fNWrite;
    if( fHashMap.size() >= fMaxCacheSize )
        Reduce( fMaxCacheSize / 2 );

    std::pair<CacheMapIterator, bool> insertion = fHashMap.insert( std::make_pair(input, result) );
    if (!insertion.second)
        insertion.first->second = result;

    KLOGGER( logger, "common.valuecache" );
    if( fReportingFrequency > 0 && fNWrite % fReportingFrequency == 0 )
        KDEBUG( logger, "VCache [" << fCacheName << "] Size: " << Size() << ", Efficiency: " << Efficiency() << ", Memory: " << Memory() << ", Avg. Load Factor: " << fHashMap.load_factor() << ", Max. Load Factor: " << fHashMap.max_load_factor() );

    return result;
}

template< class Input, class Result >
inline bool KValueCache< Input, Result >::Find( const Input& input ) const
{
    return fHashMap.find( input ) != fHashMap.end();
}

template< class Input, class Result >
inline bool KValueCache< Input, Result >::Get( const Input& input, Result& result ) const
{
    CacheMapConstIterator it = fHashMap.find( input );

    ++fNRead;

    //    KLOGGER(logger, "common.valuecache");
    //    if (fReportingFrequency > 0 && fNRead % fReportingFrequency == 0) {
    //        KDEBUG(logger, "VCache [" << fCacheName << "] Size: " << Size() << ", Efficiency: " << Efficiency() << ", Memory: " << Memory()
    //            << ", Avg. Load Factor: " << fHashMap.load_factor() << ", Max. Load Factor: " << fHashMap.max_load_factor() );
    //    }

    if( it != fHashMap.end() )
    {
        result = it->second;
        return true;
    }
    else
    {
        return false;
    }
}

template< class Input, class Result >
inline const Result& KValueCache< Input, Result >::Get( const Input& input ) const throw (KException)
{
    CacheMapConstIterator it = fHashMap.find( input );

    if (it == fHashMap.end())
        throw KException() << "Unknown hash map key.";

    return it->second;
}

template< class Input, class Result >
inline double KValueCache< Input, Result >::Efficiency() const
{
    return (fNWrite == 0) ? 0.0 : (double) fNRead / (double) fNWrite;
}

template< class Input, class Result >
inline uint64_t KValueCache< Input, Result >::Memory() const
{
    return sizeof(typename CacheMap::value_type) * fHashMap.size();
}

template< class Input, class Result >
void KValueCache< Input, Result >::EnableReporting( uint32_t reportFreq, const std::string& cacheName )
{
    fReportingFrequency = reportFreq;
    fCacheName = cacheName;
}

} /* namespace katrin */
#endif /* KVALUECACHE_H_ */
