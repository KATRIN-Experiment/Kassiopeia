#ifndef BINNING_BINNING_FACADE_H
#define BINNING_BINNING_FACADE_H

#include "BinningInterface.hpp"
#include "OstreamJoiner.h"

#include <algorithm>

namespace Binning
{

template<class DerivedBinning> class BinningFacade : public BinningInterface<DerivedBinning>
{

    friend class BinningInterface<DerivedBinning>;

  protected:
    BinningFacade() = default;
    auto numberOfEdgesImpl() const;
    auto lowerEdgeImpl() const;
    auto upperEdgeImpl() const;
    auto upperEdgeImpl(std::ptrdiff_t index) const;
    auto rangeImpl() const;
    auto widthImpl(std::ptrdiff_t index) const;
    auto centreImpl(std::ptrdiff_t index) const;
    bool validImpl() const;
    bool differImpl(const BinningFacade<DerivedBinning>& other) const;
    void printImpl(std::ostream& output) const;
};

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::numberOfEdgesImpl() const
{

    return this->numberOfBins() + 1;
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::lowerEdgeImpl() const
{

    return this->lowerEdge(0);
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::upperEdgeImpl() const
{

    return this->upperEdge(this->numberOfBins() - 1);
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::upperEdgeImpl(std::ptrdiff_t index) const
{

    return this->lowerEdge(index + 1);
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::BinningFacade::rangeImpl() const
{

    return this->upperEdge() - this->lowerEdge();
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::BinningFacade::widthImpl(std::ptrdiff_t index) const
{

    return this->upperEdge(index) - this->lowerEdge(index);
}

template<class DerivedBinning> auto BinningFacade<DerivedBinning>::BinningFacade::centreImpl(std::ptrdiff_t index) const
{

    return .5 * (this->lowerEdge(index) + this->upperEdge(index));
}

template<class DerivedBinning> bool BinningFacade<DerivedBinning>::BinningFacade::validImpl() const
{

    return std::is_sorted(this->begin(), this->end(), [](const auto& bin1, const auto& bin2) {
        return !(bin1.upperEdge() > bin2.lowerEdge());
    });
}

template<class DerivedBinning>
bool BinningFacade<DerivedBinning>::BinningFacade::differImpl(const BinningFacade<DerivedBinning>& other) const
{

    for (auto it = std::make_pair(this->begin(), other.begin()); it.first != this->end() && it.second != other.end();
         ++it.first, ++it.second)
        if (it.first->centre() != it.second->centre())
            return true;

    return false;
}

template<class DerivedBinning> void BinningFacade<DerivedBinning>::BinningFacade::printImpl(std::ostream& output) const
{

    std::copy(this->begin(), this->end(), katrin::Kommon::MakeOstreamJoiner(output, ", "));
}

}  // namespace Binning

#endif
