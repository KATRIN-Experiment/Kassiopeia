/*
 * @file   KFunctionCache.cxx
 *
 * @date   Created on: 10.08.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KFunctionCache.h"
#include "KException.h"
#include "KStringUtils.h"

#include <boost/math/special_functions/pow.hpp>

using namespace std;
using namespace boost;
using namespace boost::math;

namespace katrin {

template<size_t D, typename IndexT>
KFunctionCache<D, IndexT>::KFunctionCache(EInterpolationMethod method, uint32_t maxCacheSize, double maxLoadFactor) :
    fMethod(method),
    fCache(maxCacheSize, maxLoadFactor)
{
    for (size_t i = 0; i<D; ++i)
        ConfigureParameter(i, 1.0, 0.0);
}

template<size_t D, typename IndexT>
void KFunctionCache<D, IndexT>::ConfigureParameter(size_t iParam, double gridConstant, double centerValue,
    optional<double> lowerBound, optional<double> upperBound)
{
    assert(iParam < D);
    fParamConfigs[iParam].fGridConstant = gridConstant;
    fParamConfigs[iParam].fCenterValue = centerValue;
    fParamConfigs[iParam].fLowerBound = lowerBound;
    fParamConfigs[iParam].fUpperBound = upperBound;
}

template<size_t D, typename IndexT>
double KFunctionCache<D, IndexT>::CachedFunctionValue(const std::array<IndexT, D>& gridIndices)
{
    double result;
    if (fCache.Get(gridIndices, result)) {
        return result;
    }
    else {
        std::array<double, D> parameterValues;
        ParameterValues(gridIndices, parameterValues);
        if (fFunctionA) {
            return fCache.Store(gridIndices, fFunctionA( parameterValues ) );
        }
        else if (fFunctionV) {
            vector<double> parameterValuesVector(parameterValues.begin(), parameterValues.end());
            return fCache.Store(gridIndices, fFunctionV( parameterValuesVector ) );
        }
        else if (fFunctionP) {
            return fCache.Store(gridIndices, fFunctionP( parameterValues.begin() ) );
        }
        else if (fFunctionS) {
            return fCache.Store(gridIndices, fFunctionS( parameterValues[0] ) );
        }
        else {
            throw KFunctionCacheException() << "No source function defined.";
        }
    }
}

/**
 * The actual interpolation is performed in this function.
 * The spline polynomes are taken from arXiv:0905.3564v1.
 *
 *
 * @param paramValues
 * @return
 */
template<size_t D, typename IndexT>
double KFunctionCache<D, IndexT>::Get(const std::array<double, D>& paramValues)
{

    std::array< IndexT, D> centerGridIndices;
    std::array< double, D> gridDistances;
    GridIndices(paramValues, centerGridIndices, gridDistances);

    if (fMethod == EInterpolationMethod::Nearest) {
        std::array< IndexT, D> nearestGridIndices = centerGridIndices;
        for (size_t d = 0; d < D; ++d) {
            if (gridDistances[d] >= 0.5)
                ++nearestGridIndices[d];
        }
        return CachedFunctionValue(nearestGridIndices);
    }

    std::array< IndexT, D> startGridIndices;
    std::array< IndexT, D> endGridIndices;
    std::array< IndexT, D> currentGridIndices;

    double splineSum = 0.0;

    // set the combinations of grid coordinates to sum over
    int32_t g;
    if (fMethod == EInterpolationMethod::Spline) {
        g = 1;
    }
    else if (fMethod == EInterpolationMethod::Linear) {
        g = 0;
    }
    else
        throw KFunctionCacheException() << "No valid interpolation method.";

    const bool combinationsSet = (
        FirstCombination(centerGridIndices, 0-g, startGridIndices) &&
        LastCombination(centerGridIndices, 1+g, endGridIndices) );

    if (!combinationsSet)
        throw KFunctionCacheException() << "Out of bounds.";

    currentGridIndices = startGridIndices;
    double polynomProduct, x;
    int32_t i;

    // summation over all combinations of grid indices
    do {
        polynomProduct = 1.0;

        // product over the spline polynomes
        for (size_t j = 0; j < D; ++j) {
            i = currentGridIndices[j] - centerGridIndices[j];
            x = gridDistances[j];

            if (fMethod == EInterpolationMethod::Spline) {

                switch(i) {
                case -1 :
                    polynomProduct *= 0.5*pow<3>(x-1.0)*x*(2.0*x+1.0); break;
                case 0 :
                    polynomProduct *= -0.5*(x-1.0)*(6.0*pow<4>(x)-9.0*pow<3>(x)+2.0*x+2.0); break;
                case 1 :
                    polynomProduct *= 0.5*x*(6.0*pow<4>(x)-15.0*pow<3>(x)+9.0*pow<2>(x)+x+1.0); break;
                case 2 :
                    polynomProduct *= -0.5*(x-1.0)*pow<3>(x)*(2.0*x-3); break;
                default:
                    throw KFunctionCacheException() << "Invalid spline polynom requested.";
                }
            }
            else if (fMethod == EInterpolationMethod::Linear) {

                switch(i) {
                case 0 :
                    polynomProduct *= (1.0 - x); break;
                case 1 :
                    polynomProduct *= x; break;
                default:
                    throw KFunctionCacheException() << "Invalid spline polynom requested.";
                }
            }

            if (polynomProduct == 0.0)
                break;
        }

        if (polynomProduct == 0.0)
            continue;

        const double functionValue = CachedFunctionValue(currentGridIndices);
//            KDEBUG(currentGridIndices << "; " << gridDistances << "; " << polynomProduct << "; " << functionValue);
        splineSum += functionValue * polynomProduct;
    }
    while ( NextCombination(startGridIndices, endGridIndices, currentGridIndices) );


    return splineSum;
}


template<size_t D, typename IndexT>
void KFunctionCache<D, IndexT>::EnableReporting(uint32_t reportFreq, const std::string& cacheName)
{
    fCache.EnableReporting(reportFreq, cacheName);
}

template class KFunctionCache<1, int32_t>;
template class KFunctionCache<2, int32_t>;
template class KFunctionCache<3, int32_t>;
template class KFunctionCache<4, int32_t>;
template class KFunctionCache<5, int32_t>;
template class KFunctionCache<6, int32_t>;
template class KFunctionCache<7, int32_t>;
template class KFunctionCache<8, int32_t>;
template class KFunctionCache<9, int32_t>;
template class KFunctionCache<10, int32_t>;

template class KFunctionCache<1, int64_t>;
template class KFunctionCache<2, int64_t>;
template class KFunctionCache<3, int64_t>;
template class KFunctionCache<4, int64_t>;
template class KFunctionCache<5, int64_t>;
template class KFunctionCache<6, int64_t>;
template class KFunctionCache<7, int64_t>;
template class KFunctionCache<8, int64_t>;
template class KFunctionCache<9, int64_t>;
template class KFunctionCache<10, int64_t>;

}
