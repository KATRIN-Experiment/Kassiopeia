/**
 * @file KMathOperands.h
 *
 * @date 20.06.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KMATHOPERANDS_H_
#define KOMMON_CORE_UTILITY_KMATHOPERANDS_H_


namespace katrin
{

#define ENABLE_IF_ARITHMETIC                                                                                           \
    class = typename std::enable_if < std::is_arithmetic<typename ContainerT::const_iterator::value_type>::value &&    \
            std::is_arithmetic<OperandT>::value > ::type

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT& operator*=(ContainerT& c, OperandT o)
{
    for (auto& v : c)
        v *= o;
    return c;
}

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT& operator/=(ContainerT& c, OperandT o)
{
    for (auto& v : c)
        v /= o;
    return c;
}

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT operator*(const ContainerT& c, OperandT o)
{
    ContainerT result(c);
    result *= o;
    return result;
}

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT operator/(const ContainerT& c, OperandT o)
{
    ContainerT result(c);
    result /= o;
    return result;
}

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT operator*(OperandT o, const ContainerT& c)
{
    return c * o;
}

template<class ContainerT, class OperandT, ENABLE_IF_ARITHMETIC>
inline ContainerT operator/(OperandT o, const ContainerT& c)
{
    return c / o;
}

}  // namespace katrin

#endif /* KOMMON_CORE_UTILITY_KMATHOPERANDS_H_ */
