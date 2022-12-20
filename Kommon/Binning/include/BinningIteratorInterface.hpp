#ifndef BINNING_BINNING_ITERATOR_INTERFACE_H
#define BINNING_BINNING_ITERATOR_INTERFACE_H

#include <iostream>

namespace Binning
{

template<class DerivedIterator> class BinningIteratorInterface
{

  public:
    auto& operator*();
    auto operator->();
    DerivedIterator& operator++();
    DerivedIterator operator++(int);
    DerivedIterator& operator--();
    DerivedIterator operator--(int);
    DerivedIterator operator+(std::ptrdiff_t n) const;
    DerivedIterator& operator+=(std::ptrdiff_t n);
    DerivedIterator operator-(std::ptrdiff_t n) const;
    DerivedIterator& operator-=(std::ptrdiff_t n);
    bool isEqualTo(const BinningIteratorInterface<DerivedIterator>& other) const;

  protected:
    BinningIteratorInterface() = default;
    const DerivedIterator& derivedIterator() const;
    DerivedIterator& derivedIterator();
};

template<class DerivedIterator>
bool operator==(const BinningIteratorInterface<DerivedIterator>& it1,
                const BinningIteratorInterface<DerivedIterator>& it2);
template<class DerivedIterator>
bool operator!=(const BinningIteratorInterface<DerivedIterator>& it1,
                const BinningIteratorInterface<DerivedIterator>& it2);

template<class DerivedIterator>
const DerivedIterator& BinningIteratorInterface<DerivedIterator>::derivedIterator() const
{

    return static_cast<const DerivedIterator&>(*this);
}

template<class DerivedIterator> DerivedIterator& BinningIteratorInterface<DerivedIterator>::derivedIterator()
{

    return static_cast<DerivedIterator&>(*this);
}

template<class DerivedIterator> auto& BinningIteratorInterface<DerivedIterator>::operator*()
{

    return derivedIterator().dereference();
}

template<class DerivedIterator> auto BinningIteratorInterface<DerivedIterator>::operator->()
{

    return &(*derivedIterator());
}

template<class DerivedIterator> DerivedIterator& BinningIteratorInterface<DerivedIterator>::operator++()
{

    derivedIterator().increment();
    return derivedIterator();
}

template<class DerivedIterator> DerivedIterator BinningIteratorInterface<DerivedIterator>::operator++(int)
{

    DerivedIterator copy{derivedIterator()};
    ++*this;
    return copy;
}

template<class DerivedIterator> DerivedIterator& BinningIteratorInterface<DerivedIterator>::operator--()
{

    derivedIterator().decrement();
    return derivedIterator();
}

template<class DerivedIterator> DerivedIterator BinningIteratorInterface<DerivedIterator>::operator--(int)
{

    DerivedIterator copy{derivedIterator()};
    --*this;
    return copy;
}

template<class DerivedIterator>
DerivedIterator BinningIteratorInterface<DerivedIterator>::operator+(std::ptrdiff_t n) const
{

    DerivedIterator copy{derivedIterator()};
    return copy += n;
}

template<class DerivedIterator> DerivedIterator& BinningIteratorInterface<DerivedIterator>::operator+=(std::ptrdiff_t n)
{

    derivedIterator().advance(n);
    return derivedIterator();
}

template<class DerivedIterator>
DerivedIterator BinningIteratorInterface<DerivedIterator>::operator-(std::ptrdiff_t n) const
{

    return derivedIterator() + (-n);
}

template<class DerivedIterator> DerivedIterator& BinningIteratorInterface<DerivedIterator>::operator-=(std::ptrdiff_t n)
{

    return derivedIterator() += -n;
}

template<class DerivedIterator>
bool BinningIteratorInterface<DerivedIterator>::isEqualTo(const BinningIteratorInterface<DerivedIterator>& other) const
{

    return derivedIterator().equal(other.derivedIterator());
}

template<class DerivedIterator>
bool operator==(const BinningIteratorInterface<DerivedIterator>& it1,
                const BinningIteratorInterface<DerivedIterator>& it2)
{

    return it1.isEqualTo(it2);
}
template<class DerivedIterator>
bool operator!=(const BinningIteratorInterface<DerivedIterator>& it1,
                const BinningIteratorInterface<DerivedIterator>& it2)
{

    return !(it1 == it2);
}

}  // namespace Binning

#endif
