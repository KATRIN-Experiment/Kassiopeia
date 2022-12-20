#include "VariableBinning.hpp"

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace Binning
{

VariableBinning::edge_type VariableBinning::ConstIterator::BinProxy::lowerEdgeImpl() const
{

    return *(binningIt->incrementable);
}

VariableBinning::edge_type VariableBinning::ConstIterator::BinProxy::upperEdgeImpl() const
{

    return *std::next(binningIt->incrementable);
}

VariableBinning::ConstIterator::ConstIterator(edge_iterator_type edgeIt) : ValuedBinningIteratorFacade(edgeIt) {}

VariableBinning::ConstIterator::ConstIterator(const VariableBinning& binning) :
    VariableBinning::ConstIterator(std::cbegin(binning.edges))
{}

VariableBinning::VariableBinning(std::vector<double> edges) : edges(std::move(edges))
{

    if (!valid())
        throw std::invalid_argument("VariableBinning: invalid edges");
}

VariableBinning::VariableBinning(const std::vector<ConstantBinning>& binnings)
{

    std::size_t numberOfEdges =
        std::accumulate(std::begin(binnings), std::end(binnings), 0, [](const auto sum, const auto& binning) {
            return sum + binning.numberOfEdges();
        });

    edges.reserve(numberOfEdges);

    for (const auto& binning : binnings)
        std::transform(std::begin(binning), std::end(binning), std::back_inserter(edges), [](const auto& bin) {
            return bin.lowerEdge();
        });

    if (!binnings.empty())
        edges.push_back(binnings.back().upperEdge());

    if (!valid())
        throw std::invalid_argument("VariableBinning: invalid constant sub-binnings");
}

VariableBinning::VariableBinning(std::initializer_list<ConstantBinning> binnings) :
    VariableBinning(std::vector<ConstantBinning>(binnings))
{}

std::size_t VariableBinning::numberOfBinsImpl() const
{

    return edges.empty() ? 0 : edges.size() - 1;
}

VariableBinning::edge_type VariableBinning::lowerEdgeImpl(std::ptrdiff_t index) const
{

    return edges[index];
}

VariableBinning::ConstIterator VariableBinning::beginImpl() const
{

    return ConstIterator(std::cbegin(edges));
}

VariableBinning::ConstIterator VariableBinning::endImpl() const
{

    return ConstIterator(std::prev(std::cend(edges)));  //one bin less than edges
}

std::istream& operator>>(std::istream& input, VariableBinning& binning)
{

    std::string text;
    input >> text;

    std::vector<std::string> tokens;
    boost::split(tokens, text, boost::is_any_of(";|"));

    std::vector<ConstantBinning> constantBinnings;
    constantBinnings.reserve(tokens.size());

    std::transform(std::cbegin(tokens), std::cend(tokens), std::back_inserter(constantBinnings), [](const auto& token) {
        return boost::lexical_cast<ConstantBinning>(token);
    });

    if (!constantBinnings.empty())
        binning = VariableBinning{constantBinnings};
    else
        throw std::invalid_argument(text + " cannot be parsed to build a variable binning.");

    return input;
}

}  // namespace Binning
