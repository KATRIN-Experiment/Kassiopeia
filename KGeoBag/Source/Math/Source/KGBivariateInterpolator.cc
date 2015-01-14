#include "KGBivariateInterpolator.hh"

namespace KGeoBag
{
  bool operator<(const std::vector<KGDataPoint<2> >& lhs,
		 const std::vector<KGDataPoint<2> >& rhs)
  {
    if (lhs.empty()) return true;
    if (rhs.empty()) return false;

    return lhs[0][1] < rhs[0][1];
  }
}
