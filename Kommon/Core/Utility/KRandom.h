/**
 * @file KRandom.h
 *
 * @date 24.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 */

#ifndef KRANDOM_H_
#define KRANDOM_H_

#include "KSingleton.h"
#include "KNonCopyable.h"

#include <math.h>
#include <limits>
#include <random>
#include <type_traits>

namespace katrin {

/**
 * A Mersenne Twister random number generator, which should be used as a singleton.
 */
template<class XEngineType>
class KRandomPrototype: public KSingletonAsReference<KRandomPrototype<XEngineType>>, KNonCopyable
{
public:
    typedef XEngineType engine_type;
    typedef typename engine_type::result_type result_type;

public:
    KRandomPrototype(result_type seed = engine_type::default_seed);
    virtual ~KRandomPrototype();

    /**
     * Get the seed, the random number engine was last initialized with.
     * @return
     */
    result_type GetSeed() const { return fSeed; }

    /**
     * Set the seed on the underlying mersenne twister engine.
     * For a seed = 0, the current system time in seconds is used as seed value.
     * @param seed
     * @return
     */
    result_type SetSeed(result_type seed = engine_type::default_seed);

    /**
     * Get a reference to the underlying random number engine (mersenne twister).
     * @return
     */
    engine_type& GetEngine() { return fEngine; }

    /**
     * Get a random uniform number in the specified range.
     * @param min lower interval bound
     * @param max upper interval bound
     * @param minIncluded Include the lower bound.
     * @param maxIncluded Include the upper bound.
     *
     * @return
     */
    template<class FloatType>
    typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
    Uniform(FloatType min, FloatType max, bool minIncluded, bool maxIncluded);

    /**
     * Get a random uniform number in the specified range [min, max).
     * @param min lower interval bound
     * @param max upper interval bound
     * @return
     */
    template<class FloatType = double>
    typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
    Uniform(FloatType min = 0.0, FloatType max = 1.0);

    /**
     * Get a random uniform number from a discrete distribution [inclMin, inclMax].
     * @param inclMin
     * @param inclMax
     * @return
     */
    template<class IntegerType>
    typename std::enable_if<std::is_integral<IntegerType>::value, IntegerType>::type
    Uniform(IntegerType inclMin, IntegerType inclMax);

    /**
     * Return a boolean.
     * @param probability
     * @return True with the given probability.
     */
    template<class FloatType = double>
    bool Bool(FloatType probability = 0.5);

    /**
     * Draw from a gaussian / normal distribution.
     * @param mean
     * @param sigma
     * @return
     */
    template<class FloatType = double>
    FloatType Gauss(FloatType mean = 0.0, FloatType sigma = 1.0);

    /**
     * Draw from an exponential distribution according to exp(-t/tau).
     * @param tau
     * @return
     */
    template<class FloatType>
    inline FloatType Exponential(FloatType tau);

    /**
     * Draw an integer value from a poisson distribution.
     * @param mean
     * @return
     */
    template<class IntegerType = uint32_t>
    typename std::enable_if<std::is_integral<IntegerType>::value, IntegerType>::type
    Poisson(double mean);

    /**
     * Draw a float value (cast from integer) from a poisson distribution.
     * @param mean
     * @return
     */
    template<class FloatType = double>
    typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
    Poisson(FloatType mean);

private:
    result_type fSeed;
    engine_type fEngine;
};

template<class XEngineType>
inline KRandomPrototype<XEngineType>::KRandomPrototype(result_type seed) :
    fSeed(0),
    fEngine()
{
    SetSeed(seed);
}

template<class XEngineType>
inline KRandomPrototype<XEngineType>::~KRandomPrototype()
{ }

template<class XEngineType>
inline typename KRandomPrototype<XEngineType>::result_type KRandomPrototype<XEngineType>::SetSeed(result_type value)
{
    fSeed = (value == 0) ? std::random_device()() : value;
    fEngine.seed(fSeed);
    return fSeed;
}

template<class XEngineType>
template<class FloatType>
typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
inline KRandomPrototype<XEngineType>::Uniform(FloatType min, FloatType max, bool minIncluded, bool maxIncluded)
{
    if (minIncluded) {
        if (!maxIncluded) {
            // pass
        }
        else {
            max = std::nextafter(max, (max > min) ? std::numeric_limits<FloatType>::max() : -std::numeric_limits<FloatType>::max());
        }
    }
    else {
        if (!maxIncluded) {
            min = std::nextafter(min, (max > min) ? std::numeric_limits<FloatType>::max() : -std::numeric_limits<FloatType>::max());
        }
        else {
            std::swap(min, max);
        }
    }
    return std::uniform_real_distribution<FloatType>(min, max)(fEngine);
}

template<class XEngineType>
template<class FloatType>
typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
inline KRandomPrototype<XEngineType>::Uniform(FloatType min, FloatType max)
{
    return std::uniform_real_distribution<FloatType>(min, max)(fEngine);
}

template<class XEngineType>
template<class IntegerType>
typename std::enable_if<std::is_integral<IntegerType>::value, IntegerType>::type
inline KRandomPrototype<XEngineType>::Uniform(IntegerType inclMin, IntegerType inclMax)
{
    return std::uniform_int_distribution<IntegerType>(inclMin, inclMax)(fEngine);
}

template<class XEngineType>
template<class FloatType>
inline bool KRandomPrototype<XEngineType>::Bool(FloatType probability)
{
    return std::uniform_real_distribution<FloatType>(0.0, 1.0)(fEngine) < probability;
}

template<class XEngineType>
template<class FloatType>
inline FloatType KRandomPrototype<XEngineType>::Gauss(FloatType mean, FloatType sigma)
{
    return std::normal_distribution<FloatType>(mean, sigma)(fEngine);
}

template<class XEngineType>
template<class IntegerType>
typename std::enable_if<std::is_integral<IntegerType>::value, IntegerType>::type
inline KRandomPrototype<XEngineType>::Poisson(double mean)
{
    return std::poisson_distribution<IntegerType>(mean)(fEngine);
}

template<class XEngineType>
template<class FloatType>
typename std::enable_if<std::is_floating_point<FloatType>::value, FloatType>::type
inline KRandomPrototype<XEngineType>::Poisson(FloatType mean)
{
    if (mean > std::numeric_limits<uint64_t>::max() / 2.0)
        return Gauss<FloatType>(mean, sqrt(mean));
    else
        return (FloatType) std::poisson_distribution<uint64_t>(mean)(fEngine);
}

template<class XEngineType>
template<class FloatType>
inline FloatType KRandomPrototype<XEngineType>::Exponential(FloatType tau)
{
    return std::exponential_distribution<FloatType>(1.0/tau)(fEngine);
}

typedef KRandomPrototype<std::mt19937> KRandom;

} /* namespace katrin */

#endif /* KRANDOM_H_ */
