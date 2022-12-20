/**
 * @file RangedConvolver.h
 *
 * @date 01.11.2018
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class to store a function and convolve it with sorted discrete probability distributions over a reduced range
 */
#ifndef KMATHS_RANGED_CONVOLVER_H
#define KMATHS_RANGED_CONVOLVER_H

#include "Convolver.hpp"

#include <algorithm>

namespace KommonMath
{

template<class Kernel> class RangedConvolver
{

    Convolver<Kernel> convolver;
    double min;
    double max;

  public:
    RangedConvolver(Kernel&& kernel, double min, double max);
    template<class SortedDistribution> double density(const SortedDistribution& distribution, double xValue) const;
    template<class SortedDistribution, class Bin>
    double probability(const SortedDistribution& distribution, const Bin& bin) const;
};

template<class Kernel, class... Args> RangedConvolver<Kernel> MakeRangedConvolver(Kernel&& kernel, Args&&... args);

template<class Kernel>
RangedConvolver<Kernel>::RangedConvolver(Kernel&& kernel, double min, double max) :
    convolver(std::forward<Kernel>(kernel)),
    min(min),
    max(max)
{

    if (!(min < max))
        throw std::invalid_argument("RangedConvolver: Invalid range: min >= max");
};

template<class Kernel>
template<class SortedDistribution>
double RangedConvolver<Kernel>::density(const SortedDistribution& distribution, double xValue) const
{

    auto begin = std::find_if(std::begin(distribution),
                              std::end(distribution),
                              [min = this->min, xValue](const auto& point) { return !(point.energy < xValue + min); });
    auto end = std::find_if(begin, std::end(distribution), [max = this->max, xValue](const auto& point) {
        return !(point.energy < xValue + max);
    });
    if (begin != std::begin(distribution))
        --begin;
    if (end != std::end(distribution))
        ++end;

    return convolver.density(begin, end, xValue);
}

template<class Kernel>
template<class SortedDistribution, class Bin>
double RangedConvolver<Kernel>::probability(const SortedDistribution& distribution, const Bin& bin) const
{

    return density(distribution, bin.centre()) * bin.width();
}

template<class Kernel, class... Args> RangedConvolver<Kernel> MakeRangedConvolver(Kernel&& kernel, Args&&... args)
{

    return RangedConvolver<Kernel>(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
}

}  // namespace KommonMath

#endif
