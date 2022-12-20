/**
 * @file SmearingKernel.cxx
 *
 * @date 01.11.2018
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class defining a Gaussian (centred on zero) smearing kernel
 */

#include "SmearingKernel.hpp"

#include "boost/math/constants/constants.hpp"

#include <cmath>

namespace KommonMath
{

SmearingKernel::SmearingKernel(double sigma) : sigma(sigma) {}

double SmearingKernel::operator()(double x) const
{

    return std::exp(-.5 * std::pow(x / sigma, 2)) / (sigma * std::sqrt(2 * boost::math::constants::pi<double>()));
}

}  // namespace KommonMath
