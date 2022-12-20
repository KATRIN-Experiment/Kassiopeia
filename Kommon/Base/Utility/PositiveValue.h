#ifndef KOMMON_POSITIVE_VALUE_H
#define KOMMON_POSITIVE_VALUE_H

/**
 * @file PositiveValue.h
 * @brief Inherit from this class to enforce positive values. Provide a Name implementation.
 * @date 22.11.2018
 * @author Valerian Sibille <vsibille@mit.edu>
 */

#include "KException.h"

#include <istream>
#include <memory>

namespace katrin
{

namespace Kommon
{

template<class Quantity, class ValueType = double> class PositiveValue
{

    ValueType value;

    const Quantity& Derived() const;
    Quantity& Derived();
    bool Invalid() const;
    void ThrowIfInvalid() const;

  public:
    PositiveValue();
    explicit PositiveValue(ValueType value_);
    operator ValueType() const;
    void SetValue(ValueType value_);
    Quantity& operator+=(const Quantity& other);
    Quantity& operator+=(const ValueType& value_);

  protected:
    ~PositiveValue() = default;
};

template<class Quantity, class ValueType>
std::istream& operator>>(std::istream& input, PositiveValue<Quantity, ValueType>& positiveValue);

// Class implementation

template<class Quantity, class ValueType> const Quantity& PositiveValue<Quantity, ValueType>::Derived() const
{

    return static_cast<const Quantity&>(*this);
}

template<class Quantity, class ValueType> Quantity& PositiveValue<Quantity, ValueType>::Derived()
{

    return static_cast<Quantity&>(*this);
}

template<class Quantity, class ValueType> bool PositiveValue<Quantity, ValueType>::Invalid() const
{

    return value < 0;
}

template<class Quantity, class ValueType> void PositiveValue<Quantity, ValueType>::ThrowIfInvalid() const
{

    if (Invalid())
        throw KException() << Derived().Name() << " cannot be negative!";
}
template<class Quantity, class ValueType> PositiveValue<Quantity, ValueType>::PositiveValue() : value(0.) {}

template<class Quantity, class ValueType>
PositiveValue<Quantity, ValueType>::PositiveValue(ValueType value_) : value(std::move(value_))
{

    ThrowIfInvalid();
}

template<class Quantity, class ValueType> PositiveValue<Quantity, ValueType>::operator ValueType() const
{

    return value;
}

template<class Quantity, class ValueType> void PositiveValue<Quantity, ValueType>::SetValue(ValueType value_)
{

    value = std::move(value_);
    ThrowIfInvalid();
}

template<class Quantity, class ValueType>
Quantity& PositiveValue<Quantity, ValueType>::operator+=(const Quantity& other)
{

    value += other.value;
    return Derived();
}

template<class Quantity, class ValueType>
Quantity& PositiveValue<Quantity, ValueType>::operator+=(const ValueType& value_)
{

    value += value_;
    ThrowIfInvalid();
    return Derived();
}

// Free operators' implementations

template<class Quantity, class ValueType>
std::istream& operator>>(std::istream& input, PositiveValue<Quantity, ValueType>& positiveValue)
{

    ValueType value;
    input >> value;
    positiveValue.SetValue(std::move(value));
    return input;
}

}  // namespace Kommon

}  // namespace katrin

#endif
