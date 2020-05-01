#ifndef BINNING_CONSTANT_BINNING_H
#define BINNING_CONSTANT_BINNING_H

#include "BinProxyFacade.hpp"
#include "BinningFacade.hpp"
#include "ValuedBinningIteratorFacade.hpp"

#include <iostream>

namespace Binning
{

class ConstantBinning : public BinningFacade<ConstantBinning>
{

    friend class BinningInterface<ConstantBinning>;
    using edge_type = double;
    std::size_t numberOfBins_;
    edge_type lowerEdge_;
    edge_type width_;

  public:
    class ConstIterator :
        public ValuedBinningIteratorFacade<ConstIterator, std::ptrdiff_t, BinProxyFacade<ConstIterator>>
    {

        friend class BinningIteratorInterface<ConstIterator>;

      public:
        class BinProxy : public BinProxyFacade<ConstIterator>
        {

            friend BinInterface<ConstIterator>;

          protected:
            edge_type lowerEdgeImpl() const;
            edge_type upperEdgeImpl() const;
            edge_type widthImpl() const;
        };

        using value_type = const BinProxy;
        using pointer = const BinProxy*;
        using reference = const BinProxy&;

        explicit ConstIterator(const ConstantBinning& binning, std::ptrdiff_t index = 0);

      protected:
        bool equal(const ConstIterator& other) const;

      private:
        const ConstantBinning* binning;
    };

    class Width
    {

        edge_type width_;

      public:
        explicit Width(edge_type width);
        operator edge_type() const;
    };

    ConstantBinning();
    ConstantBinning(std::size_t numberOfBins, edge_type lowerEdge, edge_type upperEdge);
    ConstantBinning(Width maxBinWidth, edge_type lowerEdge, edge_type upperEdge);

  protected:
    std::size_t numberOfBinsImpl() const;
    edge_type lowerEdgeImpl(std::ptrdiff_t index = 0) const;
    edge_type widthImpl(std::ptrdiff_t) const;
    edge_type centreImpl(std::ptrdiff_t index) const;
    bool validImpl() const;
    ConstIterator beginImpl() const;
    ConstIterator endImpl() const;
    void printImpl(std::ostream& output) const;
};

std::istream& operator>>(std::istream& input, ConstantBinning& binning);

}  // namespace Binning

#endif
