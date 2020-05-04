/**
 * @file SmearingKernel.hpp
 *
 * @date 01.11.2018
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class defining a Gaussian (centred on zero) smearing kernel
 */
#ifndef KMATHS_SMEARING_KERNEL_H
#define KMATHS_SMEARING_KERNEL_H

namespace Maths
{

class SmearingKernel
{

    double sigma;

  public:
    SmearingKernel(double sigma);
    double operator()(double x) const;
};

}  // namespace Maths

#endif
