#ifndef KELLIPTICINTEGRALS_DEF
#define KELLIPTICINTEGRALS_DEF

namespace KEMField
{
struct KCompleteEllipticIntegral1stKind
{
    double operator()(double k) const;
};

struct KCompleteEllipticIntegral2ndKind
{
    double operator()(double k) const;
};

struct KEllipticEMinusKOverkSquared
{
    double operator()(double k) const;
};

struct KEllipticCarlsonSymmetricRC
{
    double operator()(double x, double y) const;
};

struct KEllipticCarlsonSymmetricRD
{
    double operator()(double x, double y, double z) const;
};

struct KEllipticCarlsonSymmetricRF
{
    double operator()(double x, double y, double z) const;
};

struct KEllipticCarlsonSymmetricRJ
{
    double operator()(double x, double y, double z, double p) const;
};

}  // namespace KEMField

#endif /* KELLIPTICINTEGRALS_DEF */
