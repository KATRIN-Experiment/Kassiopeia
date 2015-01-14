/*
 * @file   KValueInterpolator.h
 *
 * @date   Created on: 08.08.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KVALUEINTERPOLATOR_H_
#define KVALUEINTERPOLATOR_H_

#include "KValueCache.h"
#include "KNumeric.h"
#include "KException.h"

#include <vector>

#include <boost/enum.hpp>
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <boost/math/special_functions/modf.hpp>
#include <boost/static_assert.hpp>

namespace katrin
{

class KFunctionCacheException : public KExceptionPrototype<KFunctionCacheException, KException>
{ };

BOOST_ENUM(KEInterpolationMethod,
    (Nearest)(Linear)(Spline)
);

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
template<std::size_t D = 1, typename IndexT = int32_t>
class KFunctionCache
{
public:
    typedef KValueCache<boost::array<IndexT, D>, double> ValueCache_t;
    typedef boost::function<double(const boost::array<double, D>&)> FunctionArray_t;
    typedef boost::function<double(const std::vector<double>&)> FunctionVector_t;
    typedef boost::function<double(const double*)> FunctionPointer_t;
    typedef boost::function<double(double)> FunctionScalar_t;

public:
    KFunctionCache(KEInterpolationMethod method = KEInterpolationMethod::Spline, uint32_t maxCacheSize = 1048576,
            double maxLoadFactor = 2.0);
    ~KFunctionCache() { }

    void SetMaxCacheSize(uint32_t maxCacheSize) { fCache.SetMaxSize(maxCacheSize); }

    void SetMethod(KEInterpolationMethod method) { fMethod = method; }

    void ConfigureParameter(uint32_t iParam, double gridConstant, double centerValue = 0.0,
            boost::optional<double> lowerBound = boost::none, boost::optional<double> upperBound = boost::none);

    template<class FunctionT>
    void SetFunctionWithCArray(FunctionT& function);
    template<class FunctionT>
    void SetFunctionWithBoostArray(FunctionT& function);
    template<class FunctionT>
    void SetFunctionWithVector(FunctionT& function);
    template<class FunctionT>
    void SetFunctionWithScalar(FunctionT& function);

    static std::size_t NumberOfDimensions() { return D; }

    double Get(const boost::array<double, D>& params);
    double Get(const double* params);
    double Get(const std::vector<double>& params);
    double Get(const double& param);

    uint32_t CacheSize() const { return fCache.Size(); }
    void ClearCache() { fCache.Clear(); }
    void EnableReporting(uint32_t reportFreq = 1000, const std::string& cacheName = "");

protected:
    void GridIndex(uint32_t iParam, double paramValue, IndexT& resultIndex, double& resultDistance) const;
    void GridIndices(const boost::array<double, D>& paramValues, boost::array<IndexT, D>& resultIndices,
            boost::array<double, D>& resultDistances) const;

    bool FirstCombination(const boost::array<IndexT, D>& start, double lower,
            boost::array<IndexT, D>& result) const;
    bool LastCombination(const boost::array<IndexT, D>& start, double upper,
            boost::array<IndexT, D>& result) const;
    bool NextCombination(const boost::array<IndexT, D>& first, const boost::array<IndexT, D>& last,
            boost::array<IndexT, D>& current) const;

    double ParameterValue(uint32_t iParam, IndexT gridIndex, double gridDist = 0.0) const;
    void ParameterValues(const boost::array<IndexT, D>& gridIndices,
            boost::array<double, D>& resultValues) const;

    double CachedFunctionValue(const boost::array<IndexT, D>& gridIndices);

    KEInterpolationMethod fMethod;

    ValueCache_t fCache;

    struct ParameterConfig
    {
        double fGridConstant;
        double fCenterValue;
        boost::optional<double> fLowerBound;
        boost::optional<double> fUpperBound;
    };
    boost::array<ParameterConfig, D> fParamConfigs;

    FunctionPointer_t fFunctionP;
    FunctionArray_t fFunctionA;
    FunctionVector_t fFunctionV;
    FunctionScalar_t fFunctionS;

};

template<std::size_t D, typename IndexT>
template<class FunctionT>
inline void KFunctionCache<D, IndexT>::SetFunctionWithCArray(FunctionT& function)
{
    fFunctionV.clear();
    fFunctionA.clear();
    fFunctionS.clear();
    fFunctionP = function;
    fCache.Clear();
}

template<std::size_t D, typename IndexT>
template<class FunctionT>
inline void KFunctionCache<D, IndexT>::SetFunctionWithBoostArray(FunctionT& function)
{
    fFunctionV.clear();
    fFunctionP.clear();
    fFunctionS.clear();
    fFunctionA = function;
    fCache.Clear();
}

template<std::size_t D, typename IndexT>
template<class FunctionT>
inline void KFunctionCache<D, IndexT>::SetFunctionWithVector(FunctionT& function)
{
    fFunctionA.clear();
    fFunctionP.clear();
    fFunctionS.clear();
    fFunctionV = function;
    fCache.Clear();
}

template<std::size_t D, typename IndexT>
template<class FunctionT>
inline void KFunctionCache<D, IndexT>::SetFunctionWithScalar(FunctionT& function)
{
    BOOST_STATIC_ASSERT(D == 1);

    fFunctionA.clear();
    fFunctionP.clear();
    fFunctionV.clear();
    fFunctionS = function;
    fCache.Clear();
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const std::vector<double>& params)
{
    assert(params.size() == D);
    boost::array<double, D> barray;
    std::copy(params.begin(), params.end(), barray.begin());
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const double* params)
{
    boost::array<double, D> barray;
    std::copy(params, params + D, barray.begin());
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline double KFunctionCache<D, IndexT>::Get(const double& param)
{
    boost::array<double, D> barray;
    barray[0] = param;
    return Get(barray);
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::GridIndex(uint32_t iParam, double paramValue, IndexT& resultIndex,
        double& resultDistance) const
{
    assert(iParam < D);
    paramValue = KNumeric::Limit(paramValue, fParamConfigs[iParam].fLowerBound,
            fParamConfigs[iParam].fUpperBound);
    const double floatGridIndex = (paramValue - fParamConfigs[iParam].fCenterValue)
            / fParamConfigs[iParam].fGridConstant;

//    std::cout << iParam << ": " << paramValue << " = " << fParamConfigs[iParam].fGridConstant << " * " << floatGridIndex << " + " << fParamConfigs[iParam].fCenterValue << std::endl;

    resultDistance = boost::math::modf(floatGridIndex, &resultIndex);
    if (resultDistance < 0.0) {
        resultDistance += 1.0;
        --resultIndex;
    }
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::GridIndices(const boost::array<double, D>& paramValues,
        boost::array<IndexT, D>& resultIndices, boost::array<double, D>& resultDistances) const
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
    return KNumeric::Limit(result, fParamConfigs[iParam].fLowerBound, fParamConfigs[iParam].fUpperBound);
}

template<std::size_t D, typename IndexT>
inline void KFunctionCache<D, IndexT>::ParameterValues(const boost::array<IndexT, D>& gridIndices,
        boost::array<double, D>& resultValues) const
{
    for (std::size_t i = 0; i < D; ++i)
        resultValues[i] = ParameterValue(i, gridIndices[i]);
}

template<std::size_t D, typename IndexT>
inline bool KFunctionCache<D, IndexT>::FirstCombination(const boost::array<IndexT, D>& start,
        double lower, boost::array<IndexT, D>& result) const
{
    for (std::size_t i = 0; i < D; ++i) {
        if (start[i] < KNumeric::Min<IndexT>() - lower)
            return false;
        result[i] = start[i] + lower;
    }
    return true;
}

template<std::size_t D, typename IndexT>
inline bool KFunctionCache<D, IndexT>::LastCombination(const boost::array<IndexT, D>& start,
        double upper, boost::array<IndexT, D>& result) const
{
    for (std::size_t i = 0; i < D; ++i) {
        if (start[i] > KNumeric::Max<IndexT>() - upper)
            return false;
        result[i] = start[i] + upper;
    }
    return true;
}

template< std::size_t D, typename IndexT >
inline bool KFunctionCache<D, IndexT>::NextCombination(const boost::array<IndexT, D>& first,
        const boost::array<IndexT, D>& last, boost::array<IndexT, D>& current) const
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

}

#endif /* KVALUEINTERPOLATOR_H_ */
