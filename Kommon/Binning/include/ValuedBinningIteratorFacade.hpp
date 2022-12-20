#ifndef BINNING_VALUED_BINNING_ITERATOR_FACADE_H
#define BINNING_VALUED_BINNING_ITERATOR_FACADE_H

#include "BinningIteratorInterface.hpp"

namespace Binning
{

template<class DerivedIterator, class RandomlyIncrementable, class ValueType>
class ValuedBinningIteratorFacade : public BinningIteratorInterface<DerivedIterator>
{

    friend class BinningIteratorInterface<DerivedIterator>;

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;

  protected:
    ValuedBinningIteratorFacade(RandomlyIncrementable incrementable) : incrementable(incrementable) {}
    const auto& dereference()
    {
        bin = ValueType{this->derivedIterator()};
        return bin;
    }
    void increment()
    {
        advance(1);
    }
    void decrement()
    {
        advance(-1);
    }
    void advance(difference_type n)
    {
        incrementable += n;
    }
    bool equal(const DerivedIterator& other) const
    {
        return incrementable == other.incrementable;
    }

    RandomlyIncrementable incrementable;
    ValueType bin;
};

}  // namespace Binning

#endif
