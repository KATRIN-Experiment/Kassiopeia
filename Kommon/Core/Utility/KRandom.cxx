/**
 * @file KRandom.cxx
 *
 * @date 25.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KRandom.h"
#include <ctime>

using namespace std;

namespace {

template <typename IntType>
inline double log_factorial(IntType k)
{
    assert(k >= 0);
    assert(k < 10);

    const double poisson_hormann_table[10] = { 0.0, 0.0, 0.69314718055994529, 1.7917594692280550, 3.1780538303479458,
        4.7874917427820458, 6.5792512120101012, 8.5251613610654147, 10.604602902745251, 12.801827480081469 };

    return poisson_hormann_table[k];
}

}

namespace katrin {

KRandom::result_type KRandom::SetSeed(result_type value)
{
    if (!value)
        value = time(0);
    fSeed = value;
    seed(fSeed);
    return fSeed;
}

double KRandom::Gauss(double mean, double sigma)
{
    const double kC1 = 1.448242853;
    const double kC2 = 3.307147487;
    const double kC3 = 1.46754004;
    const double kD1 = 1.036467755;
    const double kD2 = 5.295844968;
    const double kD3 = 3.631288474;
    const double kHm = 0.483941449;
    const double kZm = 0.107981933;
    const double kHp = 4.132731354;
    const double kZp = 18.52161694;
    const double kPhln = 0.4515827053;
    const double kHm1 = 0.516058551;
    const double kHp1 = 3.132731354;
    const double kHzm = 0.375959516;
    const double kHzmp = 0.591923442;
    /*zhm 0.967882898*/

    const double kAs = 0.8853395638;
    const double kBs = 0.2452635696;
    const double kCs = 0.2770276848;
    const double kB = 0.5029324303;
    const double kX0 = 0.4571828819;
    const double kYm = 0.187308492;
    const double kS = 0.7270572718;
    const double kT = 0.03895759111;

    double result;
    double rn, x, y, z;

    do {
        y = Uniform(0.0, 1.0, false, false);

        if (y > kHm1) {
            result = kHp * y - kHp1;
            break;
        }

        else if (y < kZm) {
            rn = kZp * y - 1;
            result = (rn > 0) ? (1 + rn) : (-1 + rn);
            break;
        }

        else if (y < kHm) {
            rn = Uniform(0.0, 1.0, false, false);
            rn = rn - 1 + rn;
            z = (rn > 0) ? 2 - rn : -2 - rn;
            if ((kC1 - y) * (kC3 + fabs(z)) < kC2) {
                result = z;
                break;
            }
            else {
                x = rn * rn;
                if ((y + kD1) * (kD3 + x) < kD2) {
                    result = rn;
                    break;
                }
                else if (kHzmp - y < exp(-(z * z + kPhln) / 2)) {
                    result = z;
                    break;
                }
                else if (y + kHzm < exp(-(x + kPhln) / 2)) {
                    result = rn;
                    break;
                }
            }
        }

        while (1) {
            x = Uniform(0.0, 1.0, false, false);
            y = kYm * Uniform(0.0, 1.0, false, false);
            z = kX0 - kS * x - y;
            if (z > 0)
                rn = 2 + y / x;
            else {
                x = 1 - x;
                y = kYm - y;
                rn = -(2 + y / x);
            }
            if ((y - kAs + x) * (kCs + x) + kBs < 0) {
                result = rn;
                break;
            }
            else if (y < x + kT)
                if (rn * rn < 4 * (kB - log(x))) {
                    result = rn;
                    break;
                }
        }
    } while (0);

    return mean + sigma * result;

}

double KRandom::Exponential(double tau)
{
    return -log( Uniform(0.0, 1.0, false, true) ) * tau;
}

uint32_t KRandom::Poisson(double mean)
{
    if (mean <= 0.0)
        return 0;

    // use the PTRD algorithm:
    if (mean >= 10.0) {

        struct {
            double v_r;
            double a;
            double b;
            double smu;
            double inv_alpha;
        } ptrd;

        ptrd.smu = sqrt(mean);
        ptrd.b = 0.931 + 2.53 * ptrd.smu;
        ptrd.a = -0.059 + 0.02483 * ptrd.b;
        ptrd.inv_alpha = 1.1239 + 1.1328 / (ptrd.b - 3.4);
        ptrd.v_r = 0.9277 - 3.6224 / (ptrd.b - 2);

        while (true) {
            double u;
            double v = Uniform(0.0, 1.0, true, false);
            if (v <= 0.86 * ptrd.v_r) {
                u = v / ptrd.v_r - 0.43;
                return static_cast<uint32_t>(floor(
                        (2 * ptrd.a / (0.5 - abs(u)) + ptrd.b) * u + mean + 0.445));
            }

            if (v >= ptrd.v_r) {
                u = Uniform(0.0, 1.0, true, false) - 0.5;
            }
            else {
                u = v / ptrd.v_r - 0.93;
                u = ((u < 0) ? -0.5 : 0.5) - u;
                v = Uniform(0.0, 1.0, true, false) * ptrd.v_r;
            }

            double us = 0.5 - abs(u);
            if (us < 0.013 && v > us) {
                continue;
            }

            double k = floor((2 * ptrd.a / us + ptrd.b) * u + mean + 0.445);
            v = v * ptrd.inv_alpha / (ptrd.a / (us * us) + ptrd.b);

            double log_sqrt_2pi = 0.91893853320467267;

            if (k >= 10) {
                if (log(v * ptrd.smu)
                        <= (k + 0.5) * log(mean / k) - mean - log_sqrt_2pi + k
                                - (1 / 12. - (1 / 360. - 1 / (1260. * k * k)) / (k * k)) / k) {
                    return static_cast<uint32_t>(k);
                }
            }
            else if (k >= 0) {
                if (log(v) <= k * log(mean) - mean - log_factorial(static_cast<uint32_t>(k))) {
                    return static_cast<uint32_t>(k);
                }
            }
        }
    }

    // use inversion otherwise:

    double p = exp(-mean);
    uint32_t x = 0;
    double u = Uniform(0.0, 1.0, true, false);
    while (u > p) {
        u = u - p;
        ++x;
        p = mean * p / x;
    }
    return x;

}

double KRandom::PoissonDouble(double mean)
{
    if (mean >= std::numeric_limits<uint32_t>::max())
        return Gauss(mean, sqrt(mean));
    else
        return (double) Poisson(mean);
}

//    /**
//     * Returns an array of n random numbers uniformly distributed in (0,1].
//     */
//    void RandomArray(size_t aN, double* aArray)
//    {
//        for (size_t i = 0; i < aN; ++i) {
//            *(aArray+i) = Uniform(0.0, 1.0, false, true);
//        }
//    }

}
