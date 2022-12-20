/**
 * @file KAlgorithm.h
 *
 * @date 05.10.2017
 * @author Valerian Sibille <vsibille@mit.edu>
 */
#ifndef K_ALGORITHM_H_
#define K_ALGORITHM_H_

#include <algorithm>
#include <iterator>

namespace katrin
{

namespace Algorithm
{
// the following supports maps implemented as std::vector<std::pair> which lack map::at
template<class Map, class Key> const auto& FindValue(const Map& map, const Key& key)
{

    auto it = std::find_if(std::cbegin(map), std::cend(map), [&](const auto& pair) { return pair.first == key; });
    if (it == std::cend(map))
        throw std::out_of_range("Utility::Algorithm::FindValue(map, key): key not in map");

    return it->second;
}

template<class Map, class Value> const auto& FindKey(const Map& map, const Value& value)
{

    auto it = std::find_if(std::cbegin(map), std::cend(map), [&](const auto& pair) { return pair.second == value; });
    if (it == std::cend(map))
        throw std::out_of_range("Utility::Algorithm::FindKey(map, value): value not in map");

    return it->first;
}

template<class RandomAccessIt, class Distance>
RandomAccessIt SumStride(RandomAccessIt begin, RandomAccessIt end, Distance stride)
{
    while (begin != end) {
        *begin += *(begin + stride);
        ++begin;
    }
    return begin;
}

}  // namespace Algorithm

}  // namespace katrin

#endif
