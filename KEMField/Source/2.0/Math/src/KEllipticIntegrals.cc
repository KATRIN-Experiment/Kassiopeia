#include "KEllipticIntegrals.hh"

#include <cmath>

#ifdef KEMFIELD_USE_GSL
#include "gsl/gsl_sf_ellint.h"
#include "gsl/gsl_machine.h"
#endif

namespace KEMField
{
  /**
   * A variant of the Chebyshev approximation for complete elliptic integral K
   * (W.J.Cody, Math. of Comp. 19 (1965) 105).
   */
  double KCompleteEllipticIntegral1stKind::operator() (double k) const
  {
#ifdef KEMFIELD_USE_GSL
    if (k >= 1.) k = 1. - GSL_DBL_EPSILON;
    return gsl_sf_ellint_Kcomp(k,GSL_PREC_DOUBLE);
#else
    double eta = 1.-k*k;
    static const double ln4=1.386294361119890;
    static const double a[10]={9.657359028085625e-2,3.088514627130518e-2,
				 1.493801353268716e-2,8.789801874555064e-3,
				 6.179627446053317e-3,6.847909282624505e-3,
				 9.848929322176893e-3,8.003003980649985e-3,
				 2.296634898396958e-3,1.393087857006646e-4};
    static const double b[10]={1.249999999999080e-1,7.031249973903835e-2,
				 4.882804190686239e-2,3.737773975862360e-2,
				 3.012484901289893e-2,2.393191332311079e-2,
				 1.553094163197720e-2,5.973904299155429e-3,
				 9.215546349632498e-4,2.970028096655561e-5};
    double etan,suma,sumb;
    int n;
    suma=sumb=0.;
    etan=eta;
    for(n=0;n<10;n++)
    {
      suma+=a[n]*etan;
      sumb+=b[n]*etan;
      etan*=eta;
    }
    return ln4+suma-(1./2.+sumb)*log(fabs(eta)+1.e-19);
#endif
  }

  /**
   * Chebyshev approximation for complete elliptic integral E (W.J.Cody, Math. of
   * Comp. 19 (1965) 105).
   */
  double KCompleteEllipticIntegral2ndKind::operator() (double k) const
  {
#ifdef KEMFIELD_USE_GSL
    return gsl_sf_ellint_Ecomp(k,GSL_PREC_DOUBLE);
#else
    double eta = 1.-k*k;
    static const double c[10]={4.431471805608895e-1,5.680519456755915e-2,
				 2.183181167613048e-2,1.156959574529540e-2,
				 7.595093422559432e-3,7.820404060959554e-3,
				 1.077063503986645e-2,8.638442173604074e-3,
				 2.468503330460722e-3,1.494662175718132e-4};
    static const double d[10]={2.499999999999017e-1,9.374999972120314e-2,
				 5.859366125553149e-2,4.271789054738309e-2,
				 3.347894366576162e-2,2.614501470031387e-2,
				 1.680402334636338e-2,6.432146586438301e-3,
				 9.898332846225384e-4,3.185919565550157e-5};

    double etan,sumc,sumd;
    int n;
    sumc=sumd=0.;
    etan=eta;
    for(n=0;n<10;n++)
    {
      sumc+=c[n]*etan;
      sumd+=d[n]*etan;
      etan*=eta;
      if(etan<1.e-20) break;
    }
    return 1.+sumc-sumd*log(fabs(eta)+1.e-19);
#endif
  }

  /**
   * Computes (E_elliptic-K_elliptic)/(k*k); eta=1-k*k.
   */
  double KEllipticEMinusKOverkSquared::operator() (double k) const
  {
    static KCompleteEllipticIntegral1stKind K;
    static KCompleteEllipticIntegral2ndKind E;

    double k2,EK,k2n,a,b,cn;
    int n;
    k2=k*k;
    if(k2>0.8)
      EK=(E(k)-K(k))/k2;
    else
    {
      a=M_PI/2.;
      k2n=1.;
      EK=0.;
      for(n=1;n<=900;n++)
      {
	cn=(2.*n-1)/(2.*n);
	a*=cn*cn;
	b=-a/(2.*n-1.);
	EK+=(b-a)*k2n;
	k2n*=k2;
	if(k2n<1.e-16) break;
      }
    }
    return EK;
  }

  /**
   * Computes the Carlson symmetric elliptic integral R_{C}
   */
  double KEllipticCarlsonSymmetricRC::operator() (double x,double y) const
  {
#ifdef KEMFIELD_USE_GSL
    return gsl_sf_ellint_RC(x,y,GSL_PREC_DOUBLE);
#else
    // From Kassiopeia's Magfield3 implementation:
    //
    // This function computes Carlson's degenerate elliptic integral:
    // R_C(x,y). x must be nonnegative, and y must be nonzero.
    // If y<0, the Cauchy principal value is returned.
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.001,THIRD = 1. / 3., C1 = 0.3, C2 = 1. / 7., C3 = 0.375, C4 = 9. / 22.;
    double alamb, ave, s, w, xt, yt;
    if(y > 0.)
    {
      xt = x;
      yt = y;
      w = 1.;
    }
    else
    {
      xt = x - y;
      yt = -y;
      w = sqrt(x) / sqrt(xt);
    }
    do
    {
      alamb = 2. * sqrt(xt) * sqrt(yt) + yt;
      xt = 0.25 * (xt + alamb);
      yt = 0.25 * (yt + alamb);
      ave = THIRD * (xt + yt + yt);
      s = (yt - ave) / ave;
    }
    while(fabs(s) > ERRTOL);
    return w * (1. + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / sqrt(ave);
#endif
  }

  /**
   * Computes the Carlson symmetric elliptic integral R_{D}
   */
  double KEllipticCarlsonSymmetricRD::operator() (double x,double y,double z) const
  {
#ifdef KEMFIELD_USE_GSL
    return gsl_sf_ellint_RD(x,y,z,GSL_PREC_DOUBLE);
#else
    // From Kassiopeia's Magfield3 implementation:
    //
    // This function computes Carlson's elliptic integral of the second kind:
    // R_D(x,y,z). x and y must be nonnegative, and at most one can be zero.
    // z must be positive
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.0015, C1 = 3. / 14., C2 = 1. / 6., C3 = 9. / 22., C4 = 3. / 26., C5 = 0.25 * C3, C6 = 1.5 * C4;
    double alamb, ave, delx, dely, delz, ea, eb, ec, ed, ee, fac, sum, sqrtx, sqrty, sqrtz, xt, yt, zt;
    xt = x;
    yt = y;
    zt = z;
    sum = 0.;
    fac = 1.;
    do
    {
      sqrtx = sqrt(xt);
      sqrty = sqrt(yt);
      sqrtz = sqrt(zt);
      alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
      sum += fac / (sqrtz * (zt + alamb));
      fac = 0.25 * fac;
      xt = 0.25 * (xt + alamb);
      yt = 0.25 * (yt + alamb);
      zt = 0.25 * (zt + alamb);
      ave = 0.2 * (xt + yt + 3. * zt);
      delx = (ave - xt) / ave;
      dely = (ave - yt) / ave;
      delz = (ave - zt) / ave;
    }
    while(fabs(delx) > ERRTOL || fabs(dely) > ERRTOL || fabs(delz) > ERRTOL);
    ea = delx * dely;
    eb = delz * delz;
    ec = ea - eb;
    ed = ea - 6. * eb;
    ee = ed + ec + ec;
    return 3. * sum + fac * (1. + ed * (-C1 + C5 * ed - C6 * delz * ee) + delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea))) / (ave * sqrt(ave));
#endif
  }

  /**
   * Computes the Carlson symmetric elliptic integral R_{F}
   */
  double KEllipticCarlsonSymmetricRF::operator() (double x,double y,double z) const
  {
#ifdef KEMFIELD_USE_GSL
    return gsl_sf_ellint_RF(x,y,z,GSL_PREC_DOUBLE);
#else
    // From Kassiopeia's Magfield3 implementation:
    //
    // This function computes Carlson's elliptic integral of the first kind:
    // R_F(x,y,z). x, y, z must be nonnegative, and at most one can be zero
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.002, C1 = 1. / 24., C2 = 0.1, C3 = 3. / 44., C4 = 1. / 14., THIRD = 1. / 3.;
    double alamb, ave, delx, dely, delz, e2, e3, sqrtx, sqrty, sqrtz, xt, yt, zt;
    xt = x;
    yt = y;
    zt = z;
    do
    {
      sqrtx = sqrt(xt);
      sqrty = sqrt(yt);
      sqrtz = sqrt(zt);
      alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
      xt = 0.25 * (xt + alamb);
      yt = 0.25 * (yt + alamb);
      zt = 0.25 * (zt + alamb);
      ave = THIRD * (xt + yt + zt);
      delx = (ave - xt) / ave;
      dely = (ave - yt) / ave;
      delz = (ave - zt) / ave;
    }
    while(fabs(delx) > ERRTOL || fabs(dely) > ERRTOL || fabs(delz) > ERRTOL);
    e2 = delx * dely - delz * delz;
    e3 = delx * dely * delz;
    return (1. + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave);
#endif
  }

  /**
   * Computes the Carlson symmetric elliptic integral R_{J}
   */
  double KEllipticCarlsonSymmetricRJ::operator() (double x,double y,double z,double p) const
  {
#ifdef KEMFIELD_USE_GSL
    static const double lolim = pow(5.0 * GSL_DBL_MIN, 1.0/3.0);

    if (fabs(y)<lolim)
      y = lolim;
    if (fabs(p)<lolim)
      p = lolim;

    return gsl_sf_ellint_RJ(x,y,z,p,GSL_PREC_DOUBLE);
#else
    // From Kassiopeia's Magfield3 implementation:
    //
    // This function computes Carlson's elliptic integral of the third kind:
    // R_J(x,y,z,p). x, y and z must be nonnegative, and at most one can be
    // zero. p must be nonzero. If p<0, the Cauchy principal value is returned.
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    static KEllipticCarlsonSymmetricRC ellipticCarlsonSymmetricRC;
    static KEllipticCarlsonSymmetricRF ellipticCarlsonSymmetricRF;
    const double ERRTOL = 0.0015, C1 = 3. / 14., C2 = 1. / 3., C3 = 3. / 22., C4 = 3. / 26., C5 = 0.75 * C3, C6 = 1.5 * C4, C7 = 0.5 * C2, C8 = 2. * C3;
    double a = 0., alamb, alpha, ans, ave, b = 0., beta, delp, delx, dely, delz, ea, eb, ec, ed, ee, fac, pt, rcx = 0., rho, sum, sqrtx, sqrty, sqrtz, tau, xt, yt, zt;
    sum = 0.;
    fac = 1.;
    if(p > 0.)
    {
      xt = x;
      yt = y;
      zt = z;
      pt = p;
    }
    else
    {
      xt = x; if (y<xt) xt = y; if (z<xt) xt = z;
      zt = x; if (y>zt) zt = y; if (z>zt) zt = z;
      yt = x + y + z - xt - zt;
      a = 1. / (yt - p);
      b = a * (zt - yt) * (yt - xt);
      pt = yt + b;
      rho = xt * zt / yt;
      tau = p * pt / yt;
      rcx = ellipticCarlsonSymmetricRC(rho,tau);
    }
    do
    {
      sqrtx = sqrt(xt);
      sqrty = sqrt(yt);
      sqrtz = sqrt(zt);
      alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
      alpha = pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz;
      alpha*=alpha;
      beta = pt * (pt+alamb)*(pt+alamb);
      sum += fac * ellipticCarlsonSymmetricRC(alpha,beta);
      ;
      fac = 0.25 * fac;
      xt = 0.25 * (xt + alamb);
      yt = 0.25 * (yt + alamb);
      zt = 0.25 * (zt + alamb);
      pt = 0.25 * (pt + alamb);
      ave = 0.2 * (xt + yt + zt + 2. * pt);
      delx = (ave - xt) / ave;
      dely = (ave - yt) / ave;
      delz = (ave - zt) / ave;
      delp = (ave - pt) / ave;
    }
    while(fabs(delx) > ERRTOL || fabs(dely) > ERRTOL || fabs(delz) > ERRTOL || fabs(delp) > ERRTOL);
    ea = delx * (dely + delz) + dely * delz;
    eb = delx * dely * delz;
    ec = delp * delp;
    ed = ea - 3. * ec;
    ee = eb + 2. * delp * (ea - ec);
    ans = 3. * sum + fac * (1. + ed * (-C1 + C5 * ed - C6 * ee) + eb * (C7 + delp * (-C8 + delp * C4)) + delp * ea * (C2 - delp * C3) - C2 * delp * ec) / (ave * sqrt(ave));
    if(p < 0.)
      ans = a * (b * ans + 3. * (rcx - ellipticCarlsonSymmetricRF(xt, yt, zt)));
    return ans;
#endif
  }
}
