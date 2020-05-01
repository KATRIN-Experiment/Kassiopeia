/**
 * @file KCast.h
 *
 * @date 05.10.2017
 * @author Valerian Sibille <vsibille@mit.edu>
 */
#ifndef K_CAST_H_
#define K_CAST_H_

#include <type_traits>

namespace katrin
{

namespace Cast
{

template<class T> constexpr auto Underlying(T t) noexcept
{

    return static_cast<typename std::underlying_type<T>::type>(t);
}


}  // namespace Cast

}  // namespace katrin

#endif
