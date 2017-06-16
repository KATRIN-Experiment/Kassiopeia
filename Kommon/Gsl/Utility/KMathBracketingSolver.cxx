#include "KMathBracketingSolver.h"

namespace katrin
{

    KMathBracketingSolver::KMathBracketingSolver()
    {
    }
    KMathBracketingSolver::~KMathBracketingSolver()
    {
    }

    KMathBracketingSolver::Algorithms KMathBracketingSolver::sAlgorithms = Algorithms();

    KMathBracketingSolver::Algorithms::Algorithms()
    {
        fTypes[ eBisection ] = gsl_root_fsolver_alloc( gsl_root_fsolver_bisection );
        fTypes[ eFalsePositive ] = gsl_root_fsolver_alloc( gsl_root_fsolver_falsepos );
        fTypes[ eBrent ] = gsl_root_fsolver_alloc( gsl_root_fsolver_brent );
    }
    KMathBracketingSolver::Algorithms::~Algorithms()
    {
        gsl_root_fsolver_free( fTypes[ eBisection ] );
        gsl_root_fsolver_free( fTypes[ eFalsePositive ] );
        gsl_root_fsolver_free( fTypes[ eBrent ] );
    }

    gsl_root_fsolver* KMathBracketingSolver::Algorithms::operator[]( unsigned int aType )
    {
        return fTypes[ aType ];
    }

}
