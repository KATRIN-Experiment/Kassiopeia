/**
 * @file KVariadicOperations.h
 *
 * @date 05.10.2017
 * @author Valerian Sibille <vsibille@mit.edu>
 */
#ifndef K_VARIADIC_OPERATIONS_H_
#define K_VARIADIC_OPERATIONS_H_

#include <utility>

namespace katrin
{

namespace VariadicOperations
{

constexpr auto Product()
{

    return 1;
}

template<class First, class... Others> constexpr auto Product(First&& first, Others&&... others)
{

    return first * Product(std::forward<Others>(others)...);
}

}  // namespace VariadicOperations

}  // namespace katrin

#endif
