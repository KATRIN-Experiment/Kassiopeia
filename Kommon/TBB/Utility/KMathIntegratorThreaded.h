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
struct ThreadedSumming;
struct ThreadedKahanSumming;
}

/**
 * A parallel version of KMathIntegrator's summing policy, using Intel's Thread Building Blocks
 * library to parallelize the iterative evaluation of the sampling points.
 */
template<class Float>
using KMathIntegratorThreaded = KMathIntegrator<Float, policies::ThreadedSumming>;
template<class Float>
using KMathIntegrator2DThreaded = KMathIntegrator2D<Float, policies::ThreadedSumming>;

namespace policies {

  struct ThreadedSumming
  {
      template<class Float, class XIntegrandType, class Sum = Float>
      static Float SumSamplingPoints(uint32_t n, Float xStart, Float del, XIntegrandType&& integrand)
      {
          // don't parallelize for only n <= 2 sampling points
          if (n < 3) {

              Sum sum = 0.0;
              for (uint32_t j = 0; j < n; ++j) sum += integrand(xStart + j * del);
              return sum;

          }
          else return tbb::parallel_reduce(tbb::blocked_range<uint32_t>( 0, n ), 0.,
              [&](const tbb::blocked_range<uint32_t>& range, Sum value){
                  for (uint32_t j = range.begin(); j != range.end(); ++j) value += integrand(xStart + j * del);
                  return value;
              },
              std::plus<Float>()
          );
      }
  };
  
  struct ThreadedKahanSumming : KahanSumming
  {
      template<class Float, class XIntegrandType>
      static Float SumSamplingPoints(uint32_t n, Float xStart, Float del, XIntegrandType&& integrand)
      {
  		return ThreadedSumming::SumSamplingPoints<Float, XIntegrandType, KMathKahanSum<Float>>(n, xStart, del, std::forward<XIntegrandType>(integrand));
      }
  };

}

} /* namespace katrin */
#endif /* KMATHINTEGRATORTHREADED_H_ */
