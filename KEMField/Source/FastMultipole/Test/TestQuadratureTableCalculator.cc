#include <iostream>
#include <cmath>
#include <iomanip>

#include <cmath>

#include "KFMMessaging.hh"
#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"

#include "KFMGaussLegendreQuadratureTableCalculator.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    KFMGaussLegendreQuadratureTableCalculator* calc = new KFMGaussLegendreQuadratureTableCalculator();

    std::vector<double> w;
    std::vector<double> x;

    std::cout<< std::setprecision(16);

    unsigned int n;

    for(unsigned int n_terms = 1; n_terms < 22; n_terms++)
    {
        n = n_terms;
        calc->SetNTerms(n);
        calc->Initialize();

        calc->GetWeights(&w);
        calc->GetAbscissa(&x);

        std::cout<<"( Weights, Abscissa) = "<<std::endl;
        for(unsigned int i = 0; i<n; i++)
        {
            std::cout<<"("<<w[i]<<", "<<x[i]<<")"<<std::endl;
        }
        std::cout<<std::endl;
    }

    return 0;
}
