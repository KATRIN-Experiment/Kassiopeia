/**
 * @file Gaussian.hpp
 *
 * @date 05.03.2019
 * @author Valérian Sibille <vsibille@mit.edu>
 * @brief Class defining a Gaussian
 */
#ifndef KMATHS_GAUSSIAN_H
#define KMATHS_GAUSSIAN_H

namespace KommonMath
{

class Gaussian
{

    double mean;
    double sigma;
    bool Invalid() const;
    void ThrowIfInvalid() const;

  public:
    Gaussian(double mean, double sigma);
    double operator()(double x) const;
    double GetMean() const;
    double GetSigma() const;
    void SetMean(double mean);
    void SetSigma(double sigma);
};

}  // namespace KommonMath

#endif
