#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMGaussLegendreQuadratureTableCalculator.hh"
#include "KVMNumericalIntegrator.hh"
#include "KVMField.hh"

namespace KEMField
{

#define DIM 3

class KVMFieldTest: public KVMField
{
    public:
        KVMFieldTest(){;};
        ~KVMFieldTest(){;};

        unsigned int GetNDimDomain() const {return DIM;};
        unsigned int GetNDimRange() const {return 1;};

        void Evaluate(const double* in, double* out) const
        {
            double r2 =0;
            for(unsigned int i=0; i<DIM; i++){r2 += in[i]*in[i];};
            out[0] = std::exp(-0.5*r2);
        }
};

}

using namespace KEMField;



int main(int /*argc*/, char** /*argv*/)
{
    std::cout<< std::setprecision(15);

    unsigned int n_quad = 10;

    KVMNumericalIntegrator<DIM,1> numInt;
    KVMFieldTest integrand;

    KFMGaussLegendreQuadratureTableCalculator calc;
    calc.SetNTerms(n_quad);
    calc.Initialize();

    std::vector<double> w;
    std::vector<double> x;

    calc.GetWeights(&w);
    calc.GetAbscissa(&x);

    numInt.SetNTerms(n_quad);
    numInt.SetWeights(&(w[0]));
    numInt.SetAbscissa(&(x[0]));

    double low[DIM];
    double up[DIM];
    double result[1];
    for(unsigned int i=0; i<DIM; i++)
    {
        low[i] = -2.0;
        up[i] = 2.0;
    }

    numInt.SetLowerLimits(low);
    numInt.SetUpperLimits(up);

    numInt.SetIntegrand(&integrand);

    numInt.Integral(result);

    double exact = 1.0;
    double factor = 2.3925760266452164;

    unsigned int n_eval = 1;
    for(unsigned int i=0; i<DIM; i++)
    {
        exact *= factor;
        n_eval *= n_quad;
    }

    std::cout<<"number of integrand evaluations = "<<n_eval<<std::endl;

    std::cout<<"numerical result = "<<result[0]<<std::endl;
    std::cout<<"exact resutl = "<<exact<<std::endl;

    std::cout<<"absolute error = "<<std::abs(result[0] - exact)<<std::endl;
    std::cout<<"relative error = "<<std::abs( (result[0] - exact)/exact )<<std::endl;

    return 0;
}
