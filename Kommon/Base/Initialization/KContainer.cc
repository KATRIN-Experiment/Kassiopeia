#include "KContainer.hh"

namespace katrin
{

KContainer::KContainer() : KNamed(), fHolder(nullptr) {}
KContainer::~KContainer()
{
    if (fHolder != nullptr) {
        delete fHolder;
        fHolder = nullptr;
    }
}

bool KContainer::Empty() const
{
    if (fHolder != nullptr) {
        return false;
    }
    return true;
}

}  // namespace katrin
