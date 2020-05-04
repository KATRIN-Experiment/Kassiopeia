#include "KGaussianQuadrature.hh"

#include <cmath>

namespace KEMField
{
/**
   * A variation of Dr. Ferenc Glueck's numerical integration routine.
   *
   * @param f A pointer to an array of pointers to functions to be integrated.
   * @param m Number of functions to be integrated.
   * @param a Lower limit of integration.
   * @param b Upper limit of integration.
   * @param par A pointer to an array of parameters needed by functions.
   * @param n Granularity of numerical integrator.
   * @param ans An array of the results from integration.
   */
void KGaussianQuadrature::operator()(double (**f)(const double*, double*), int m, double a, double b, double* par,
                                     int n, double* ans) const
{
    int i, j;
    double deln;
    static const double w5[6] = {0.3187500000000000e+00,
                                 0.1376388888888889e+01,
                                 0.6555555555555556e+00,
                                 0.1212500000000000e+01,
                                 0.9256944444444445e+00,
                                 0.1011111111111111e+01};
    static const double w9[10] = {0.2803440531305107e0,
                                  0.1648702325837748e1,
                                  -0.2027449845679092e0,
                                  0.2797927414021179e1,
                                  -0.9761199294532843e0,
                                  0.2556499393738999e1,
                                  0.1451083002645404e0,
                                  0.1311227127425048e1,
                                  0.9324249063051143e0,
                                  0.1006631393298060e1};
    if (n < 12)
        n = 12;  // n too small
    deln = (b - a) / n;

    for (j = 0; j < m; j++)
        ans[j] = 0;

    double x;

    if (n < 20) {
        for (i = 0; i < 6; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += w5[i] * f[j](&x, par);
        }
        for (i = 6; i < n - 5; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += f[j](&x, par);
        }
        for (i = n - 5; i <= n; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += w5[n - i] * f[j](&x, par);
        }
    }
    else {
        for (i = 0; i < 10; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += w9[i] * f[j](&x, par);
        }
        for (i = 10; i < n - 9; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += f[j](&x, par);
        }
        for (i = n - 9; i <= n; i++) {
            x = a + deln * i;
            for (j = 0; j < m; j++)
                ans[j] += w9[n - i] * f[j](&x, par);
        }
    }

    for (j = 0; j < m; j++)
        ans[j] *= deln;

    return;
}
}  // namespace KEMField
