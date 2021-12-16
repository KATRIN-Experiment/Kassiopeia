/**
 * @file KMakeArray.h
 *
 * @date 08.10.2017
 * @author Valerian Sibille <vsibille@mit.edu>
 */
#ifndef K_MAKE_ARRAY_H_
#define K_MAKE_ARRAY_H_

#include <array>

namespace katrin
{

namespace Experimental
{

template<typename... T>
constexpr auto MakeArray(T&&... values)
    -> std::array<typename std::decay<typename std::common_type<T...>::type>::type, sizeof...(T)>
{
    return std::array<typename std::decay<typename std::common_type<T...>::type>::type, sizeof...(T)>{
        {std::forward<T>(values)...}};
}

}  // namespace Experimental

}  // namespace katrin

#endif
