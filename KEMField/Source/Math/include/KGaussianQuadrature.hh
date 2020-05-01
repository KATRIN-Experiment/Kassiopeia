#ifndef KGAUSSIANQUADRATURE_DEF
#define KGAUSSIANQUADRATURE_DEF


namespace KEMField
{
struct KGaussianQuadrature
{
    void operator()(double (**f)(const double*, double*), int m, double a, double b, double* par, int n,
                    double* ans) const;
};
}  // namespace KEMField

#endif /* KGAUSSIANQUADRATURE_DEF */
