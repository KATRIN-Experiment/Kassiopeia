/*
 * @file   KMathIntegratorThreaded.h
 *
 * @date   Created on: 08.09.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KMATHINTEGRATORTHREADED_H_
#define KMATHINTEGRATORTHREADED_H_

#include "KMathIntegrator.h"
#include "KMathIntegrator2D.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace katrin
{

// forward declarations
namespace integrator_policies {
struct ThreadedPlainSumming;
struct ThreadedKahanSumming;
}

/**
 * A parallel version of KMathIntegrator's summing policy, using Intel's Thread Building Blocks
 * library to parallelize the iterative evaluation of the sampling points.
 */
using KMathIntegratorThreaded = KMathIntegratorPrototype<integrator_policies::ThreadedKahanSumming>;
using KMathIntegrator2DThreaded = KMathIntegrator2DPrototype<integrator_policies::ThreadedKahanSumming>;

namespace integrator_policies {

struct ThreadedPlainSumming
{
    template<class XIntegrandType>
    double SumSamplingPoints(uint32_t n, const double& xStart, const double& del, XIntegrandType& integrand) const
    {
        // don't parallelize for only n <= 2 sampling points
        if (n <= 2) {
            double sum = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                const double x = xStart + (double) j * del;
                sum += integrand(x);
            }
            return sum;
        }

        return tbb::parallel_reduce(
        tbb::blocked_range<uint32_t>( 0, n ),
            0.0,
            [&](const tbb::blocked_range<uint32_t>& r, double value)->double {
                for (uint32_t j = r.begin(); j != r.end(); ++j) {
                    const double x = xStart + (double) j * del;
                    value += integrand(x);
                }
                return value;
            },
            std::plus<double>()
        );
    }
};

struct ThreadedKahanSumming : KahanSumming
{
    template<class XIntegrandType>
    double SumSamplingPoints(uint32_t n, const double& xStart, const double& del, XIntegrandType& integrand) const
    {
        // don't parallelize for only n <= 2 sampling points
        if (n <= 2) {
            KMathKahanSum sum;
            for (uint32_t j = 0; j < n; j++) {
                const double x = xStart + (double) j * del;
                sum += integrand(x);
            }
            return sum;
        }

        return tbb::parallel_reduce(
        tbb::blocked_range<uint32_t>( 0, n ),
            KMathKahanSum(),
            [&](const tbb::blocked_range<uint32_t>& r, KMathKahanSum value)->KMathKahanSum {
                for (uint32_t j = r.begin(); j != r.end(); ++j) {
                    const double x = xStart + (double) j * del;
                    value += integrand(x);
                }
                return value;
            },
            std::plus<KMathKahanSum>()
        );
    }
};

}

} /* namespace katrin */
#endif /* KMATHINTEGRATORTHREADED_H_ */
