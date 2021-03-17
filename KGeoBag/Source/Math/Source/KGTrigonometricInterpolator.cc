#include "KGTrigonometricInterpolator.hh"

#include <cmath>

namespace KGeoBag
{
KGTrigonometricInterpolator::KGTrigonometricInterpolator() : fOrder(3), fXMin(0.), fXMax(2. * M_PI) {}

void KGTrigonometricInterpolator::Initialize(DataSet& data)
{
    fA.resize(fOrder + 1, 0.);
    fB.resize(fOrder + 1, 0.);

    // fXMin = fXMax = data[0][0];
    // for (unsigned int i=0;i<data.size();i++)
    // {
    //   if (fXMin > data[i][0]) fXMin = data[i][0];
    //   if (fXMax < data[i][0]) fXMax = data[i][0];
    // }

    double y_i, x_i;

    for (unsigned int k = 0; k <= fOrder; k++) {
        fA[k] = fB[k] = 0.;
        for (auto& i : data) {
            x_i = i[0];
            y_i = i[1];

            for (unsigned int j = 0; j < k; j++)
                y_i -= fA[j] * sin(j * x_i) + fB[j] * cos(j * x_i);

            fA[k] += y_i * sin(k * x_i);
            fB[k] += y_i * cos(k * x_i);
        }
        fA[k] *= ((k != 0 ? 2. : 1.) / data.size());
        fB[k] *= ((k != 0 ? 2. : 1.) / data.size());
    }
}

int KGTrigonometricInterpolator::OutOfRange(double x) const
{
    if (x < fXMin)
        return -1;
    if (x > fXMax)
        return 1;
    return 0;
}

double KGTrigonometricInterpolator::Range(unsigned int i) const
{
    return (i == 0 ? fXMin : fXMax);
}

double KGTrigonometricInterpolator::operator()(double x) const
{
    double y = fB[0];
    for (unsigned int i = 1; i <= fOrder; i++)
        y += fA[i] * sin(i * x) + fB[i] * cos(i * x);
    return y;
}
}  // namespace KGeoBag
