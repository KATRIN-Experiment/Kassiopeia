#include "ConstantBinning.hpp"

#include "boost/algorithm/string.hpp"

#include <iomanip>
#include <vector>

namespace Binning
{

ConstantBinning::edge_type ConstantBinning::ConstIterator::BinProxy::lowerEdgeImpl() const
{

    return binningIt->binning->lowerEdge(binningIt->incrementable);
}

ConstantBinning::edge_type ConstantBinning::ConstIterator::BinProxy::upperEdgeImpl() const
{

    return binningIt->binning->upperEdge(binningIt->incrementable);
}

ConstantBinning::edge_type ConstantBinning::ConstIterator::BinProxy::widthImpl() const
{

    return binningIt->binning->width();
}

ConstantBinning::ConstIterator::ConstIterator(const ConstantBinning& binning, std::ptrdiff_t index) :
    ValuedBinningIteratorFacade(index),
    binning(&binning)
{}

bool ConstantBinning::ConstIterator::equal(const ConstIterator& other) const
{

    return ValuedBinningIteratorFacade<ConstIterator, std::ptrdiff_t, BinProxyFacade<ConstIterator>>::equal(other) &&
           binning == other.binning;
}

ConstantBinning::Width::Width(ConstantBinning::edge_type width) : width_(width)
{

    if (!(width_ > 0))
        throw std::invalid_argument("ConstantBinning: negative maximum bin width");
}

ConstantBinning::Width::operator ConstantBinning::edge_type() const
{

    return width_;
}

ConstantBinning::ConstantBinning() : ConstantBinning(1, 0, 1) {}

ConstantBinning::ConstantBinning(std::size_t numberOfBins, edge_type lowerEdge, edge_type upperEdge) :
    numberOfBins_(numberOfBins),
    lowerEdge_(lowerEdge),
    width_(numberOfBins ? (upperEdge - lowerEdge) / numberOfBins : 0)
{

    if (!valid())
        throw std::invalid_argument("ConstantBinning: invalid arguments");
}

ConstantBinning::ConstantBinning(Width maxBinWidth, edge_type lowerEdge, edge_type upperEdge)
{

    std::size_t numberOfBins = (upperEdge - lowerEdge) / maxBinWidth;
    if (numberOfBins * maxBinWidth < (upperEdge - lowerEdge))
        ++numberOfBins;

    *this = ConstantBinning(numberOfBins, lowerEdge, upperEdge);
}

std::size_t ConstantBinning::numberOfBinsImpl() const
{

    return numberOfBins_;
}

ConstantBinning::edge_type ConstantBinning::lowerEdgeImpl(std::ptrdiff_t index) const
{

    return lowerEdge_ + index * width();
}

ConstantBinning::edge_type ConstantBinning::widthImpl(std::ptrdiff_t) const
{

    return width_;
}

ConstantBinning::edge_type ConstantBinning::centreImpl(std::ptrdiff_t index) const
{

    return lowerEdge(index) + .5 * width();
}

bool ConstantBinning::validImpl() const
{

    return this->numberOfBins() && this->lowerEdge() < this->upperEdge();
}

ConstantBinning::ConstIterator ConstantBinning::beginImpl() const
{

    return ConstIterator(*this);
}

ConstantBinning::ConstIterator ConstantBinning::endImpl() const
{

    return ConstIterator(*this, numberOfBins());
}

void ConstantBinning::printImpl(std::ostream& output) const
{

    output << std::setw(5) << std::left << numberOfBins() << " [ " << std::setw(6) << std::right << lowerEdge() << ", "
           << std::setw(6) << std::right << upperEdge() << " ]";
}

std::istream& operator>>(std::istream& input, ConstantBinning& binning)
{

    std::string text;
    input >> text;

    std::vector<std::string> tokens;
    boost::split(tokens, text, boost::is_any_of(",:"));

    if (tokens.size() == 3)
        binning =
            ConstantBinning{static_cast<std::size_t>(std::stod(tokens[0])), std::stod(tokens[1]), std::stod(tokens[2])};
    else
        throw std::invalid_argument(text + " cannot be parsed to build a constant binning.");

    return input;
}

}  // namespace Binning
