/*
 * @file   KValueInterpolator.h
 *
 * @date   Created on: 08.08.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KVALUEINTERPOLATOR_H_
#define KVALUEINTERPOLATOR_H_

#include "KException.h"
#include "KHashMap.h"

#include <vector>
#include <array>
#include <functional>

#include <boost/math/special_functions/modf.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/optional.hpp>

namespace katrin
{

class KFunctionCacheException : public KExceptionPrototype<KFunctionCacheException, KException>
{ };

/**
 * A multi-variate mathematical value / function interpolator.
 * To set up the interpolator (of fixed number of dimensions), you have to specify a function, an
 * interpolation method and the grid constants for each function parameter.
 * The interpolator will then dynamically (on demand) evaluate the function on those grid points, store the
 * values in a hash map and use them to calculate interpolations at points between the grid points.
 *
 * @tparam D The number of dimensions.
 *
 */
template<size_t D = 1, typename IndexT = int32_t>
class KFunctionCache
{
public:
    enum class EInterpolationMethod {
        Nearest, Linear, Spline
    };

    typedef KHashMap<std::array<IndexT, D>, double, katrin::hash_container<std::array<IndexT, D>> > ValueCache_t;
    typedef std::function<double(const std::array<double, D>&)> Function_t;

public:
    KFunctionCache(EInterpolationMethod method = EInterpolationMethod::Spline, uint32_t maxCacheSize = 1048576,
            double maxLoadFactor = 2.0);
    ~KFunctionCache() { }

    void SetMaxCacheSize(size_t maxCacheSize) { fCache.SetMaxSize(maxCacheSize); }

    void SetMethod(EInterpolationMethod method) { fMethod = method; }

    void ConfigureParameter(size_t iParam, double gridConstant, double centerValue = 0.0,
            boost::optional<double> lowerBound = boost::none, boost::optional<double> upperBound = boost::none);

    template<class FunctionT>
    void SetFunction(FunctionT& function);

    static size_t NumberOfDimensions() { return D; }

    double Get(const std::array<double, D>& params);
    double Get(const double* params);
    double Get(const std::vector<double>& params);
    double Get(const double& param);

    size_t CacheSize() const { return fCache.Size(); }
    void ClearCache() { fCache.Clear(); }
    void EnableReporting(uint32_t reportFreq = 1000, const std::string& cacheName = "");

protected:
    void GridIndex(uint32_t iParam, double paramValue, IndexT& resultIndex, double& resultDistance) const;
    void GridIndices(const std::array<double, D>& paramValues, std::array<IndexT, D>& resultIndices,
            std::array<double, D>& resultDistances) const;

    bool FirstCombination(const std::array<IndexT, D>& start, double lower,
            std::array<IndexT, D>& result) const;
    bool LastCombination(const std::array<IndexT, D>& start, double upper,
            std::array<IndexT, D>& result) const;
    bool NextCombination(const std::array<IndexT, D>& first, const std::array<IndexT, D>& last,
            std::array<IndexT, D>& current) const;

    double ParameterValue(uint32_t iParam, IndexT gridIndex, double gridDist = 0.0) const;
    void ParameterValues(const std::array<IndexT, D>& gridIndices,
            std::array<double, D>& resultValues) const;

    double CachedFunctionValue(const std::array<IndexT, D>& gridIndices);

    EInterpolationMethod fMethod;

    ValueCache_t fCache;

    struct ParameterConfig {
        void Limit(double& input) const;
        double fGridConstant;
        double fCenterValue;
        boost::optional<double> fLowerBound;
        boost::optional<double> fUpperBound;
    };
    std::array<ParameterConfig, D> fParamConfigs;

    Function_t fFunction;

};

template<std::size_t D, typename IndexT>
template<class FunctionT>
inline void KFunctionCache<D, IndexT>::SetFunction(FunctionT& function)
{
    fFunction = function;
    fCache.Clear();
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const std::vector<double>& params)
{
    assert(params.size() == D);
    std::array<double, D> barray;
    std::copy(params.begin(), params.end(), barray.begin());
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const double* params)
{
    std::array<double, D> barray;
    std::copy(params, params + D, barray.begin());
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const double& param)
{
    std::array<double, D> barray;
    barray[0] = param;
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::ParameterConfig::Limit(double& input) const
{
    if (fLowerBound && input < fLowerBound.get())
        input = fLowerBound.get();
    else if (fUpperBound && input > fUpperBound.get())
        input = fUpperBound.get();
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::GridIndex(uint32_t iParam, double paramValue, IndexT& resultIndex,
        double& resultDistance) const
{
    assert(iParam < D);
    fParamConfigs[iParam].Limit(paramValue);
    const double floatGridIndex = (paramValue - fParamConfigs[iParam].fCenterValue)
            / fParamConfigs[iParam].fGridConstant;

    resultDistance = boost::math::modf(floatGridIndex, &resultIndex);
    if (resultDistance < 0.0) {
        resultDistance += 1.0;
        --resultIndex;
    }
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::GridIndices(const std::array<double, D>& paramValues,
        std::array<IndexT, D>& resultIndices, std::array<double, D>& resultDistances) const
{
    for (std::size_t i = 0; i < D; ++i)
        GridIndex(i, paramValues[i], resultIndices[i], resultDistances[i]);
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::ParameterValue(uint32_t iParam, IndexT gridIndex,
        double gridDist) const
{
    assert(iParam < D);
    double result = fParamConfigs[iParam].fCenterValue
            + fParamConfigs[iParam].fGridConstant * ((double) gridIndex + gridDist);
    fParamConfigs[iParam].Limit(result);
    return result;
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::ParameterValues(const std::array<IndexT, D>& gridIndices,
        std::array<double, D>& resultValues) const
{
    for (std::size_t i = 0; i < D; ++i)
        resultValues[i] = ParameterValue(i, gridIndices[i]);
}

template<std::size_t D, typename IndexT>
inline bool KFunctionCache<D, IndexT>::FirstCombination(const std::array<IndexT, D>& start,
        double lower, std::array<IndexT, D>& result) const
{
    for (std::size_t i = 0; i < D; ++i) {
        if (start[i] < std::numeric_limits<IndexT>::min() - lower)
            return false;
        result[i] = start[i] + lower;
    }
    return true;
}

template<std::size_t D, typename IndexT>
inline bool KFunctionCache<D, IndexT>::LastCombination(const std::array<IndexT, D>& start,
        double upper, std::array<IndexT, D>& result) const
{
    for (std::size_t i = 0; i < D; ++i) {
        if (start[i] > std::numeric_limits<IndexT>::max() - upper)
            return false;
        result[i] = start[i] + upper;
    }
    return true;
}

template< std::size_t D, typename IndexT >
inline bool KFunctionCache<D, IndexT>::NextCombination(const std::array<IndexT, D>& first,
        const std::array<IndexT, D>& last, std::array<IndexT, D>& current) const
{
    for (std::size_t i = 0; i < D; ++i) {
        if (current[i] < first[i]) {
            return false;
        }
        else if (current[i] < last[i]) {
            ++current[i];
            return true;
        }
        else if (i == D - 1 || current[i] > last[i]) {
            return false;
        }
        else { // (current[i] == last[i] && i > 0)
            current[i] = first[i];
            continue;
        }
    }
    return false; // this line should never be reached.
}

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
    boost::optional<double> lowerBound, boost::optional<double> upperBound)
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
        if (!fFunction)
        	throw KFunctionCacheException() << "No source function defined.";
        return fCache.Store(gridIndices, fFunction( parameterValues ) );
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
                    polynomProduct *= 0.5*boost::math::pow<3>(x-1.0)*x*(2.0*x+1.0); break;
                case 0 :
                    polynomProduct *= -0.5*(x-1.0)*(6.0*boost::math::pow<4>(x)-9.0*boost::math::pow<3>(x)+2.0*x+2.0); break;
                case 1 :
                    polynomProduct *= 0.5*x*(6.0*boost::math::pow<4>(x)-15.0*boost::math::pow<3>(x)+9.0*boost::math::pow<2>(x)+x+1.0); break;
                case 2 :
                    polynomProduct *= -0.5*(x-1.0)*boost::math::pow<3>(x)*(2.0*x-3); break;
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

}

#endif /* KVALUEINTERPOLATOR_H_ */
