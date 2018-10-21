/**
 * @file KHash.h
 *
 * @date 13.06.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KHASH_H_
#define KOMMON_CORE_UTILITY_KHASH_H_

#include <functional>
#include <type_traits>
#include <tuple>
#include <vector>

namespace katrin {

template<class T>
inline std::size_t hash_value(T const& v)
{
    return std::hash<T>()(v);
}

template<class T>
inline void hash_combine(std::size_t& seed, T const& v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class It>
inline void hash_range(std::size_t& seed, It first, It last)
{
    for(; first != last; ++first) {
        hash_combine(seed, *first);
    }
}

template<typename ContainerT>
struct hash_container
{
    std::size_t operator()(const ContainerT& arr) const
    {
        std::size_t seed = 0;
        hash_range(seed, arr.begin(), arr.end());
        return seed;
    }
};



template<class E>
struct hash_enum
{
    using utype = typename std::underlying_type<E>::type;
    std::size_t operator()(const E& e) const {
        return std::hash<utype>()( (utype) e );
    }
};

// Recursive template code derived from Matthieu M.
template<class Tuple, std::size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
    static void apply(std::size_t& seed, Tuple const& tuple)
    {
        HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template<class Tuple>
struct HashValueImpl<Tuple, 0>
{
    static void apply(std::size_t& seed, Tuple const& tuple)
    {
        hash_combine(seed, std::get<0>(tuple));
    }
};

}


// hash specializations for some STL containers and std::tuple

namespace std {

template<typename ... TT>
struct hash<tuple<TT...> >
{
    size_t operator()(tuple<TT...> const& tt) const
    {
        size_t seed = 0;
        katrin::HashValueImpl<tuple<TT...> >::apply(seed, tt);
        return seed;
    }

};

template<typename KeyT, typename ValueT>
struct hash<pair<KeyT, ValueT> >
{
    size_t operator()(const pair<KeyT, ValueT>& pair) const
    {
        size_t seed = 0;
        katrin::hash_combine(seed, pair.first);
        katrin::hash_combine(seed, pair.second);
        return seed;
    }
};

}


#endif /* KOMMON_CORE_UTILITY_KHASH_H_ */
