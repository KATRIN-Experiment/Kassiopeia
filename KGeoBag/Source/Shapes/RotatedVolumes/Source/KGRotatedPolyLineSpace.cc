#include "KGRotatedPolyLineSpace.hh"

#include "KGRotatedPolyLineSurface.hh"

namespace KGeoBag
{

template<> KGRotatedPolyLineSpace::Visitor::Visitor() = default;

template<> KGRotatedPolyLineSpace::Visitor::~Visitor() = default;

}  // namespace KGeoBag
