/**
 * @file Gaussian.cpp
 *
 * @date 05.03.2019
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class defining a Gaussian
 */

#include "Gaussian.hpp"

#include "boost/math/constants/constants.hpp"

#include <cmath>

namespace Maths
{

bool Gaussian::Invalid() const
{

    return !(sigma > 0);
}

void Gaussian::ThrowIfInvalid() const
{

    if (Invalid())
        throw std::invalid_argument("Invalid Gaussian!");
}

Gaussian::Gaussian(double mean, double sigma) : mean(mean), sigma(sigma)
{

    ThrowIfInvalid();
}

double Gaussian::operator()(double x) const
{

    return std::exp(-.5 * std::pow((x - mean) / sigma, 2)) /
           (sigma * std::sqrt(2 * boost::math::constants::pi<double>()));
}

double Gaussian::GetMean() const
{

    return mean;
}

double Gaussian::GetSigma() const
{

    return sigma;
}

void Gaussian::SetMean(double mean)
{

    this->mean = mean;
}

void Gaussian::SetSigma(double sigma)
{

    this->sigma = sigma;
    ThrowIfInvalid();
}

}  // namespace Maths
