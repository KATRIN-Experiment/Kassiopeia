/**
 * @file KMathsConvolver.h
 *
 * @date 01.11.2018
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class to store a function and convolve it with discrete probability distributions
 */
#ifndef KMATHS_CONVOLVER_H
#define KMATHS_CONVOLVER_H

#include <iterator>
#include <numeric>

namespace Maths
{

template<class Kernel> class Convolver
{

    Kernel kernel;

  public:
    Convolver(Kernel&& kernel);
    template<class DistributionIterator>
    double density(DistributionIterator begin, DistributionIterator end, double xValue) const;
    template<class Distribution> double density(const Distribution& distribution, double xValue) const;
    template<class DistributionIterator, class Bin>
    double probability(DistributionIterator begin, DistributionIterator end, const Bin& bin) const;
    template<class Distribution, class Bin> double probability(const Distribution& distribution, const Bin& bin) const;
};

template<class Kernel> Convolver<Kernel> MakeConvolver(Kernel&& kernel);

template<class Kernel>
Convolver<Kernel>::Convolver(Kernel&& kernel) :
    kernel(std::forward<Kernel>(kernel)){

    };

template<class Kernel>
template<class DistributionIterator>
double Convolver<Kernel>::density(DistributionIterator begin, DistributionIterator end, double xValue) const
{

    return std::accumulate(begin, end, 0., [&](double sum, const auto& point) {
        return sum + point.probability * kernel(xValue - point.energy);
    });
}

template<class Kernel>
template<class Distribution>
double Convolver<Kernel>::density(const Distribution& distribution, double xValue) const
{

    return density(std::begin(distribution), std::end(distribution), xValue);
}

template<class Kernel>
template<class DistributionIterator, class Bin>
double Convolver<Kernel>::probability(DistributionIterator begin, DistributionIterator end, const Bin& bin) const
{

    return density(begin, end, bin.centre()) * bin.width();
}

template<class Kernel>
template<class Distribution, class Bin>
double Convolver<Kernel>::probability(const Distribution& distribution, const Bin& bin) const
{

    return probability(std::begin(distribution), std::end(distribution), bin);
}

template<class Kernel> Convolver<Kernel> MakeConvolver(Kernel&& kernel)
{

    return Convolver<Kernel>(std::forward<Kernel>(kernel));
}

}  // namespace Maths

#endif
