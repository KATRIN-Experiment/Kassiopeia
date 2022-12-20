#ifndef BINNING_BIN_PROXY_FACADE_H
#define BINNING_BIN_PROXY_FACADE_H

#include "BinInterface.hpp"

namespace Binning
{

template<class DerivedIterator> class BinProxyFacade : public BinInterface<DerivedIterator>
{

    friend BinInterface<DerivedIterator>;

  public:
    BinProxyFacade() = default;
    explicit BinProxyFacade(const DerivedIterator& binningIt) : binningIt(&binningIt) {}

  protected:
    auto widthImpl() const
    {
        return this->upperEdge() - this->lowerEdge();
    }
    auto centreImpl() const
    {
        return .5 * (this->lowerEdge() + this->upperEdge());
    };
    void printImpl(std::ostream& output) const
    {
        output << "[" << this->lowerEdge() << ", " << this->upperEdge() << "]";
    };

    const DerivedIterator* binningIt;
};

}  // namespace Binning

#endif
