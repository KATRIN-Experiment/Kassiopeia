#ifndef KGBIVARIATEINTERPOLATOR_HH_
#define KGBIVARIATEINTERPOLATOR_HH_

#include <algorithm>

#include "KGInterpolator.hh"

namespace KGeoBag
{
  bool operator<(const std::vector<KGDataPoint<2> >& lhs,
		 const std::vector<KGDataPoint<2> >& rhs);

  template<class XInterpolator, class YInterpolator>
  class KGBivariateInterpolator
  {
  public:
    typedef KGDataPoint<2> DataPoint;
    typedef std::vector<DataPoint> DataSubset;
    typedef std::vector<DataSubset> DataSet;

    KGBivariateInterpolator();
    virtual ~KGBivariateInterpolator() {}

    void Initialize(std::vector<double>&,
		    std::vector<double>&,
		    std::vector<double>&);

    void Initialize(DataSet&);

    int OutOfRange(double x,double y) const;
    double XRange(unsigned int i) const;
    double YRange(unsigned int i) const;

    double operator()(double x, double y) const;

    std::vector<XInterpolator>& XInterpolators() { return fXInterpolators; }
    XInterpolator& GetXInterpolator() { return fXInterpolators[0]; }
    YInterpolator& GetYInterpolator() { return fYInterpolator; }

  private:
    std::vector<XInterpolator> fXInterpolators;

    mutable YInterpolator fYInterpolator;

    mutable KGInterpolator::DataSet fSubData;
  };

  template<class XInterpolator, class YInterpolator>
  KGBivariateInterpolator<XInterpolator,YInterpolator>::KGBivariateInterpolator() : fXInterpolators(1,XInterpolator())
  {

  }

  template<class XInterpolator, class YInterpolator>
  void KGBivariateInterpolator<XInterpolator,YInterpolator>::Initialize(std::vector<double>& x,std::vector<double>& y,std::vector<double>& z)
  {
    // Converts two-dimensional data that is defined on a grid into interpolable
    // data.  The vectors are defined so that, with vector lengths of N_x, N_y
    // and N_z respectively, N_z = N_x*N_y, and f(x_i,y_j) = z_k, where
    // k = j*N_y + i.

    DataSet data(x.size(),DataSubset(y.size()));

    std::vector<double>::iterator y_it = y.begin();
    std::vector<double>::iterator z_it = z.begin();
    DataSet::iterator set_it = data.begin();

    for (;y_it!=y.end();++y_it,++set_it)
    {
      std::vector<double>::iterator x_it = x.begin();
      DataSubset::iterator subset_it = (*set_it).begin();
      for (;x_it!=x.end();++x_it,++z_it,++subset_it)
      {
	(*subset_it)[0] = *x_it;
	(*subset_it)[1] = *y_it;
	(*subset_it)[2] = *z_it;
      }
      std::sort((*set_it).begin(),(*set_it).end());
    }

    std::sort(data.begin(),data.end());

    Initialize(data);
  }

  template<class XInterpolator, class YInterpolator>
  void KGBivariateInterpolator<XInterpolator,YInterpolator>::Initialize(DataSet& data)
  {
    fSubData.clear(); fSubData.resize(data.size());
    // Each instance of the X interpolator is a copy of the first, default
    // version.  This way, any parameters set for the X interpolator will carry
    // through to the rest.
    fXInterpolators.resize(data.size(),fXInterpolators[0]);

    for (unsigned int i=0;i<data.size();i++)
    {
      KGInterpolator::DataSet set(data[i].size());
      for (unsigned int j=0;j<data[i].size();j++)
      {
	fSubData[i][0] += data[i][j][1];
	set[j][0] = data[i][j][0];
	set[j][1] = data[i][j][2];
      }
      fSubData[i][0]/=data[i].size();

      fXInterpolators[i].Initialize(set);
    }

    // Initilaize the Y interpolator so methods that might need it can be used.
    for (unsigned int i=0;i<fSubData.size();i++)
      fSubData[i][1] = fXInterpolators[i](data[0][0][0]);

    fYInterpolator.Initialize(fSubData);
  }

  template<class XInterpolator, class YInterpolator>
  int KGBivariateInterpolator<XInterpolator,YInterpolator>::OutOfRange(double x,double y) const
  {
    return 10*fXInterpolators[0].OutOfRange(x) + fYInterpolator.OutOfRange(y);
  }

  template<class XInterpolator, class YInterpolator>
  double KGBivariateInterpolator<XInterpolator,YInterpolator>::XRange(unsigned int i) const
  {
    return fXInterpolators[0].Range(i);
  }

  template<class XInterpolator, class YInterpolator>
  double KGBivariateInterpolator<XInterpolator,YInterpolator>::YRange(unsigned int i) const
  {
    return fYInterpolator.Range(i);
  }

  template<class XInterpolator, class YInterpolator>
  double KGBivariateInterpolator<XInterpolator,YInterpolator>::operator()(double x, double y) const
  {
    for (unsigned int i=0;i<fSubData.size();i++)
      fSubData[i][1] = fXInterpolators[i](x);

    fYInterpolator.Initialize(fSubData);

    return fYInterpolator(y);
  }
}

#endif
