/**
 * @file KMathShepardInterpolator.h
 *
 * @date 12.12.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KMATHSHEPARDINTERPOLATOR_H_
#define KMATHSHEPARDINTERPOLATOR_H_

#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>

namespace katrin {

/**
 * Shepard interpolation as described in Numerical Recipes.
 */
template<int NDIM = 1>
class KMathShepardInterpolator {
public:
    KMathShepardInterpolator(double p = 2.0) : fP(p) { }

    void AddValue(const std::vector<double>& point, double value);
    void AddValue(const double* point, double value);
    void AddValue(double point, double value);

    void Reset() { fPoints.clear(); fValues.clear(); }

    size_t Size() const { return fPoints.size(); }
    bool Empty() const { return fPoints.empty(); }

    double Calculate(const std::vector<double>& point) const;
    double Calculate(const double* point) const;
    double Calculate(double point) const;

    double operator()(const std::vector<double>& point) const { return Calculate(point); }
    double operator()(const double* point) const { return Calculate(point); }
    double operator()(double point) const { return Calculate(point); }

private:
    double Interpolate(const double* pt) const;
    // Squared euclidian distance
    static double Rad2(const double* p1, const double* p2);

    double fP;
    std::vector<std::array<double, NDIM> > fPoints;
    std::vector<double> fValues;
};

template<int NDIM>
inline void KMathShepardInterpolator<NDIM>::AddValue(const std::vector<double>& point, double value)
{
    assert(point.size() == NDIM);
    AddValue(&point[0], value);
}

template<int NDIM>
inline void KMathShepardInterpolator<NDIM>::AddValue(const double* point, double value)
{
    assert(point != 0);
    std::array<double, NDIM> newPoint;
    std::copy(point, point+NDIM, newPoint.begin());
    fPoints.push_back(newPoint);
    fValues.push_back(value);
}

template<int NDIM>
inline void KMathShepardInterpolator<NDIM>::AddValue(double point, double value)
{
    assert(NDIM == 1);
    AddValue(&point, value);
}

template<int NDIM>
inline double KMathShepardInterpolator<NDIM>::Calculate(const std::vector<double>& point) const
{
    assert(NDIM == point.size());
    return Interpolate(&point[0]);
}

template<int NDIM>
inline double KMathShepardInterpolator<NDIM>::Calculate(const double* point) const
{
    assert(point != 0);
    return Interpolate(point);
}

template<int NDIM>
inline double KMathShepardInterpolator<NDIM>::Calculate(double point) const
{
    assert(NDIM == 1);
    return Interpolate(&point);
}

template<int NDIM>
inline double KMathShepardInterpolator<NDIM>::Interpolate(const double* pt) const
{
    assert(fPoints.size() == fValues.size());

    double r2, w, sum = 0., sumw = 0.;
    for (size_t i = 0; i < fPoints.size(); i++) {
        r2 = Rad2(pt, &fPoints[i][0]);
        if (r2 == 0.0)
            return fValues[i];
        w = (fP == 2.0) ? (1.0/r2) : pow( r2, -fP/2.0);
        sum += w;
        sumw += w * fValues[i];
    }

    return sumw / sum;
}

template<int NDIM>
inline double KMathShepardInterpolator<NDIM>::Rad2(const double* p1, const double* p2)
{
    double diff, sum = 0.0;
    for (int i = 0; i < NDIM; i++) {
        diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

}

#endif /* KMATHSHEPARDINTERPOLATOR_H_ */
