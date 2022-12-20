#ifndef BINNING_VARIABLE_BINNING_H
#define BINNING_VARIABLE_BINNING_H

#include "BinningFacade.hpp"
#include "ConstantBinning.hpp"

#include <vector>

namespace Binning
{

class VariableBinning : public BinningFacade<VariableBinning>
{

    friend class BinningInterface<VariableBinning>;
    std::vector<double> edges;
    using edge_iterator_type = decltype(std::cbegin(edges));
    using edge_type = std::remove_reference<decltype(*std::begin(edges))>::type;

  public:
    class ConstIterator :
        public ValuedBinningIteratorFacade<ConstIterator, edge_iterator_type, BinProxyFacade<ConstIterator>>
    {

      public:
        class BinProxy : public BinProxyFacade<ConstIterator>
        {

            friend BinInterface<ConstIterator>;

          protected:
            edge_type lowerEdgeImpl() const;
            edge_type upperEdgeImpl() const;
        };

        using value_type = const BinProxy;
        using pointer = const BinProxy*;
        using reference = const BinProxy&;

        explicit ConstIterator(edge_iterator_type edgeIt);
        explicit ConstIterator(const VariableBinning& binning);
    };

    VariableBinning() = default;
    VariableBinning(std::vector<double> edges);
    VariableBinning(const std::vector<ConstantBinning>& binnings);
    VariableBinning(std::initializer_list<ConstantBinning> binnings);

  protected:
    std::size_t numberOfBinsImpl() const;
    edge_type lowerEdgeImpl(std::ptrdiff_t index) const;
    ConstIterator beginImpl() const;
    ConstIterator endImpl() const;
};

std::istream& operator>>(std::istream& input, VariableBinning& binning);

}  // namespace Binning

#endif
