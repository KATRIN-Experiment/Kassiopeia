/**
 * @file KTypeTraits.h
 *
 * @date 02.07.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KTYPETRAITS_H_
#define KOMMON_CORE_UTILITY_KTYPETRAITS_H_

#include <type_traits>

namespace std
{

template<typename T> struct is_container
{
    typedef typename std::remove_const<T>::type test_type;

    template<typename A>
    static constexpr bool test(A* pt, A const* cpt = nullptr, decltype(pt->begin())* = nullptr,
                               decltype(pt->end())* = nullptr, decltype(cpt->begin())* = nullptr,
                               decltype(cpt->end())* = nullptr,

                               decltype(pt->clear())* = nullptr, decltype(pt->size())* = nullptr,

                               typename A::iterator* pi = nullptr, typename A::const_iterator* pci = nullptr,
                               typename A::value_type* /*pv*/ = nullptr)
    {

        using iterator = typename A::iterator;
        using const_iterator = typename A::const_iterator;
        using value_type = typename A::value_type;

        return (std::is_same<decltype(pt->begin()), iterator>::value ||
                std::is_same<decltype(pt->begin()), const_iterator>::value) &&
               (std::is_same<decltype(pt->end()), iterator>::value ||
                std::is_same<decltype(pt->end()), const_iterator>::value) &&
               std::is_same<decltype(cpt->begin()), const_iterator>::value &&
               std::is_same<decltype(cpt->end()), const_iterator>::value &&
               (std::is_same<decltype(**pi), value_type&>::value ||
                std::is_same<decltype(**pi), value_type const&>::value) &&
               std::is_same<decltype(**pci), value_type const&>::value;
    }

    template<typename A> static constexpr bool test(...)
    {
        return false;
    }

    static const bool value = test<test_type>(nullptr);
};

}  // namespace std

#endif /* KOMMON_CORE_UTILITY_KTYPETRAITS_H_ */
