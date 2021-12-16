#ifndef BINNING_BIN_INTERFACE_H
#define BINNING_BIN_INTERFACE_H

#include <iostream>

namespace Binning
{

template<class DerivedIterator> class BinInterface
{

    auto& derivedBin() const;

  public:
    auto lowerEdge() const;
    auto upperEdge() const;
    auto width() const;
    auto centre() const;
    void print(std::ostream& output) const;
};

template<class DerivedIterator>
std::ostream& operator<<(std::ostream& output, const BinInterface<DerivedIterator>& bin);

template<class DerivedIterator> auto& BinInterface<DerivedIterator>::derivedBin() const
{

    return static_cast<typename DerivedIterator::reference>(*this);
}

template<class DerivedIterator> auto BinInterface<DerivedIterator>::lowerEdge() const
{

    return derivedBin().lowerEdgeImpl();
}

template<class DerivedIterator> auto BinInterface<DerivedIterator>::upperEdge() const
{

    return derivedBin().upperEdgeImpl();
}

template<class DerivedIterator> auto BinInterface<DerivedIterator>::width() const
{

    return derivedBin().widthImpl();
}

template<class DerivedIterator> auto BinInterface<DerivedIterator>::centre() const
{

    return derivedBin().centreImpl();
}

template<class DerivedIterator> void BinInterface<DerivedIterator>::print(std::ostream& output) const
{

    derivedBin().printImpl(output);
}

template<class DerivedIterator> std::ostream& operator<<(std::ostream& output, const BinInterface<DerivedIterator>& bin)
{

    bin.print(output);
    return output;
}
}  // namespace Binning

#endif
