#include "KGInterpolator.hh"

#include <algorithm>

namespace KGeoBag
{
  void KGInterpolator::Initialize(std::vector<double>& x,std::vector<double>& y)
  {
    DataSet data(x.size());

    std::vector<double>::iterator x_it = x.begin();
    std::vector<double>::iterator y_it = y.begin();
    DataSet::iterator data_it = data.begin();

    for (;x_it!=x.end();++x_it,++y_it,++data_it)
    {
      (*data_it)[0] = *x_it;
      (*data_it)[1] = *y_it;
    }

    std::sort(data.begin(),data.end());

    Initialize(data);
  }
}
