#ifndef BINNING_BINNING_INTERFACE_H
#define BINNING_BINNING_INTERFACE_H

#include <cstddef>
#include <iostream>

namespace Binning
{

template<class DerivedBinning> class BinningInterface
{

    const DerivedBinning& derivedBinning() const;
    DerivedBinning& derivedBinning();

  public:
    auto numberOfBins() const;
    auto numberOfEdges() const;
    auto lowerEdge() const;
    auto upperEdge() const;
    auto lowerEdge(std::ptrdiff_t index) const;
    auto upperEdge(std::ptrdiff_t index) const;
    auto range() const;
    auto width(std::ptrdiff_t index = -1) const;
    auto centre(std::ptrdiff_t index) const;
    bool valid() const;
    bool differ(const BinningInterface<DerivedBinning>& other) const;
    auto size() const;
    auto begin();
    auto end();
    auto begin() const;
    auto end() const;
    auto cbegin() const;
    auto cend() const;
    bool empty() const;
    auto front() const;
    auto back() const;
    void print(std::ostream& output) const;

  protected:
    BinningInterface() = default;
};

template<class DerivedBinning>
std::ostream& operator<<(std::ostream& output, const BinningInterface<DerivedBinning>& binning);
template<class DerivedBinning>
bool operator!=(const BinningInterface<DerivedBinning>& binning1, const BinningInterface<DerivedBinning>& binning2);
template<class DerivedBinning>
bool operator==(const BinningInterface<DerivedBinning>& binning1, const BinningInterface<DerivedBinning>& binning2);

template<class DerivedBinning> const DerivedBinning& BinningInterface<DerivedBinning>::derivedBinning() const
{

    return static_cast<const DerivedBinning&>(*this);
}

template<class DerivedBinning> DerivedBinning& BinningInterface<DerivedBinning>::derivedBinning()
{

    return static_cast<DerivedBinning&>(*this);
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::numberOfBins() const
{

    return derivedBinning().numberOfBinsImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::numberOfEdges() const
{

    return derivedBinning().numberOfEdgesImpl();
    ;
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::lowerEdge() const
{

    return derivedBinning().lowerEdgeImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::upperEdge() const
{

    return derivedBinning().upperEdgeImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::lowerEdge(std::ptrdiff_t index) const
{

    return derivedBinning().lowerEdgeImpl(index);
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::upperEdge(std::ptrdiff_t index) const
{

    return derivedBinning().upperEdgeImpl(index);
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::range() const
{

    return derivedBinning().rangeImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::width(std::ptrdiff_t index) const
{

    return derivedBinning().widthImpl(index);
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::centre(std::ptrdiff_t index) const
{

    return derivedBinning().centreImpl(index);
}

template<class DerivedBinning> bool BinningInterface<DerivedBinning>::valid() const
{

    return derivedBinning().validImpl();
}

template<class DerivedBinning>
bool BinningInterface<DerivedBinning>::differ(const BinningInterface<DerivedBinning>& other) const
{

    return derivedBinning().differImpl(other.derivedBinning());
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::size() const
{

    return numberOfBins();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::begin()
{

    return derivedBinning().beginImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::end()
{

    return derivedBinning().endImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::begin() const
{

    return derivedBinning().beginImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::end() const
{

    return derivedBinning().endImpl();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::cbegin() const
{

    return begin();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::cend() const
{

    return end();
}

template<class DerivedBinning> bool BinningInterface<DerivedBinning>::empty() const
{

    return end() == begin();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::front() const
{

    return *begin();
}

template<class DerivedBinning> auto BinningInterface<DerivedBinning>::back() const
{

    return *(end() - 1);
}

template<class DerivedBinning> void BinningInterface<DerivedBinning>::print(std::ostream& output) const
{

    derivedBinning().printImpl(output);
}

template<class DerivedBinning>
bool operator!=(const BinningInterface<DerivedBinning>& binning1, const BinningInterface<DerivedBinning>& binning2)
{

    return binning1.differ(binning2);
}

template<class DerivedBinning>
bool operator==(const BinningInterface<DerivedBinning>& binning1, const BinningInterface<DerivedBinning>& binning2)
{

    return !(binning1 != binning2);
}

template<class DerivedBinning>
std::ostream& operator<<(std::ostream& output, const BinningInterface<DerivedBinning>& binning)
{

    binning.print(output);
    return output;
}

}  // namespace Binning

#endif
