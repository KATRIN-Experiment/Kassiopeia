#include "KFMResponseKernel_3DLaplaceM2L.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    auto* kernel = new KFMResponseKernel_3DLaplaceM2L();

    double source[3] = {0., 0., 0.};
    double target[3] = {1.124, 4.455, -2.565};
    int degree = 1;
    std::complex<double> res;

    //kernel->SetDegree(degree);
    kernel->SetSourceOrigin(source);
    kernel->SetTargetOrigin(target);

    int tsi, ssi;
    for (int j = 0; j <= degree; j++) {
        for (int k = -j; k <= j; k++) {
            tsi = j * (j + 1) + k;
            for (int n = 0; n <= degree; n++) {
                for (int m = -n; m <= n; m++) {
                    ssi = n * (n + 1) + m;
                    if (kernel->IsPhysical(ssi, tsi)) {
                        res = kernel->GetResponseFunction(ssi, tsi);
                        std::cout << "R(" << j << ", " << k << ", " << n << ", " << m << ") = " << res << std::endl;
                    }
                }
            }
        }
    }

    return 0;
}
