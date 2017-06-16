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
namespace policies {
struct ThreadedPlainSumming;
struct ThreadedKahanSumming;
}

/**
 * A parallel version of KMathIntegrator's summing policy, using Intel's Thread Building Blocks
 * library to parallelize the iterative evaluation of the sampling points.
 */
template<class XFloatT>
using KMathIntegratorThreaded = KMathIntegrator<XFloatT, policies::ThreadedPlainSumming>;
template<class XFloatT>
using KMathIntegrator2DThreaded = KMathIntegrator2D<XFloatT, policies::ThreadedPlainSumming>;

namespace policies {

struct ThreadedPlainSumming
{
    template<class XFloatT, class XIntegrandType>
    XFloatT SumSamplingPoints(uint32_t n, const XFloatT& xStart, const XFloatT& del, XIntegrandType& integrand) const
    {
        // don't parallelize for only n <= 2 sampling points
        if (n <= 2) {
            XFloatT sum = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                const XFloatT x = xStart + (XFloatT) j * del;
                sum += integrand(x);
            }
            return sum;
        }

        return tbb::parallel_reduce(
        tbb::blocked_range<uint32_t>( 0, n ),
            0.0,
            [&](const tbb::blocked_range<uint32_t>& r, XFloatT value)->XFloatT {
                for (uint32_t j = r.begin(); j != r.end(); ++j) {
                    const XFloatT x = xStart + (XFloatT) j * del;
                    value += integrand(x);
                }
                return value;
            },
            std::plus<XFloatT>()
        );
    }
};

struct ThreadedKahanSumming : KahanSumming
{
    template<class XFloatT, class XIntegrandType>
    XFloatT SumSamplingPoints(uint32_t n, const XFloatT& xStart, const XFloatT& del, XIntegrandType& integrand) const
    {
        // don't parallelize for only n <= 2 sampling points
        if (n <= 2) {
            KMathKahanSum<XFloatT> sum;
            for (uint32_t j = 0; j < n; j++) {
                const XFloatT x = xStart + (XFloatT) j * del;
                sum += integrand(x);
            }
            return sum;
        }

        return tbb::parallel_reduce(
        tbb::blocked_range<uint32_t>( 0, n ),
            KMathKahanSum<XFloatT>(),
            [&](const tbb::blocked_range<uint32_t>& r, KMathKahanSum<XFloatT> value)->KMathKahanSum<XFloatT> {
                for (uint32_t j = r.begin(); j != r.end(); ++j) {
                    const XFloatT x = xStart + (XFloatT) j * del;
                    value += integrand(x);
                }
                return value;
            },
            std::plus<KMathKahanSum<XFloatT>>()
        );
    }
};

}

} /* namespace katrin */
#endif /* KMATHINTEGRATORTHREADED_H_ */
