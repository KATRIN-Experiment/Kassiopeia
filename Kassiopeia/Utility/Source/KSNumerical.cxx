#include "KSNumerical.h"
#include "KTwoVector.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KThreeMatrix.hh"

#include <string>

using namespace std;
using namespace KGeoBag;

namespace Kassiopeia
{

    template< >
    struct KSNumerical< KTwoVector >
    {
        static KTwoVector Maximum() { return KTwoVector( KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum() ); }
        static KTwoVector Zero()    { return KTwoVector( KSNumerical< double >::Zero(), KSNumerical< double >::Zero() ); }
        static KTwoVector Minimum() { return KTwoVector( KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum() ); }
    };

    template< >
    struct KSNumerical< KThreeVector >
    {
        static KThreeVector Maximum() { return KThreeVector( KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum() ); }
        static KThreeVector Zero()    { return KThreeVector( KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero() ); }
        static KThreeVector Minimum() { return KThreeVector( KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum() ); }
    };

    template< >
    struct KSNumerical< KTwoMatrix >
    {
        static KTwoMatrix Maximum() { return KTwoMatrix( KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum() ); }
        static KTwoMatrix Zero()    { return KTwoMatrix( KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero() ); }
        static KTwoMatrix Minimum() { return KTwoMatrix( KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum() ); }
    };

    template< >
    struct KSNumerical< KThreeMatrix >
    {
        static KThreeMatrix Maximum() { return KThreeMatrix( KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum(), KSNumerical< double >::Maximum() ); }
        static KThreeMatrix Zero()    { return KThreeMatrix( KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero(), KSNumerical< double >::Zero() ); }
        static KThreeMatrix Minimum() { return KThreeMatrix( KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum(), KSNumerical< double >::Minimum() ); }
    };
}
