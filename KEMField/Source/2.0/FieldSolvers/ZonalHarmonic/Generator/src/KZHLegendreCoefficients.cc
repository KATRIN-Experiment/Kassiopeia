#include "KZHLegendreCoefficients.hh"

#include <cstddef>
#include <stddef.h>

namespace KEMField
{
  KZHLegendreCoefficients* KZHLegendreCoefficients::fZHLegendreCoefficients = NULL;

  KZHLegendreCoefficients* KZHLegendreCoefficients::GetInstance()
  {
    if (fZHLegendreCoefficients == 0)
      fZHLegendreCoefficients = new KZHLegendreCoefficients();
    return fZHLegendreCoefficients;
  }

  /**
   * Initializes recursive coefficients for computing SourcePoint coefficients.
   */
  void KZHLegendreCoefficients::InitializeLegendrePolynomialArrays(int coeff_num)
  {
    if (c.at(0).size()>=(unsigned int)coeff_num)
      return;

    int existingSize = 0;
    if (c.size()>0)
      existingSize = c.at(0).size();

    for (int i=0;i<12;i++)
      c.at(i).resize(coeff_num,0);

    int m,M,Mp,A,Ap,B,Bp,C,Cp;

    for(int n=(2 > existingSize ? 2 : existingSize);n<coeff_num;n++)
    {
      c.at(0).at(n)=(2.*n-1.)/(1.*n);
      c.at(1).at(n)=(n-1.)/(1.*n);
      c.at(2).at(n)=(2.*n-1.)/(1.*(n-1.));
      c.at(3).at(n)=(1.*n)/(1.*(n-1.));
      c.at(4).at(n)=(1.)/(n*1.);
      c.at(5).at(n)=(1.)/(n+1.);

      if(n>=6)
      {
	m=n-2;
	M=(m+1.)*(m+2.)*(2.*m-1.);
	Mp=(m)*(m+1.)*(2.*m-1.);
	A=(2.*m-1.)*(2.*m+1.)*(2.*m+3.);
	Ap=A;
	B=2.*m*m*(2.*m+3.)-1.;
	Bp=2.*m*(m+2.)*(2.*m-1.)-3.;
	C=m*(m-1.)*(2.*m+3.);
	Cp=m*(m+1.)*(2.*m+3.);
	c.at(6).at(n)=A/M;
	c.at(7).at(n)=B/M;
	c.at(8).at(n)=C/M;
	c.at(9).at(n)=Ap/Mp;
	c.at(10).at(n)=Bp/Mp;
	c.at(11).at(n)=Cp/Mp;
      }
    }

  }
}
