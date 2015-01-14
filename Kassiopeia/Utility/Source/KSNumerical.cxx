#include "KSNumerical.h"

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoMatrix.hh"
using KGeoBag::KTwoMatrix;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

#include <string>
using std::string;

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    template< >
    const bool KSNumerical< bool >::Maximum = true;
    template< >
    const bool KSNumerical< bool >::Zero = false;
    template< >
    const bool KSNumerical< bool >::Minimum = false;

    template< >
    const unsigned char KSNumerical< unsigned char >::Maximum = numeric_limits< unsigned char >::max();
    template< >
    const unsigned char KSNumerical< unsigned char >::Zero = 0;
    template< >
    const unsigned char KSNumerical< unsigned char >::Minimum = numeric_limits< unsigned char >::min();

    template< >
    const char KSNumerical< char >::Maximum = numeric_limits< char >::max();
    template< >
    const char KSNumerical< char >::Zero = 0;
    template< >
    const char KSNumerical< char >::Minimum = numeric_limits< unsigned char >::min();

    template< >
    const unsigned short KSNumerical< unsigned short >::Maximum = numeric_limits< unsigned short >::max();
    template< >
    const unsigned short KSNumerical< unsigned short >::Zero = 0;
    template< >
    const unsigned short KSNumerical< unsigned short >::Minimum = numeric_limits< unsigned short >::min();

    template< >
    const short KSNumerical< short >::Maximum = numeric_limits< short >::max();
    template< >
    const short KSNumerical< short >::Zero = 0;
    template< >
    const short KSNumerical< short >::Minimum = numeric_limits< unsigned short >::min();

    template< >
    const unsigned int KSNumerical< unsigned int >::Maximum = numeric_limits< unsigned int >::max();
    template< >
    const unsigned int KSNumerical< unsigned int >::Zero = 0;
    template< >
    const unsigned int KSNumerical< unsigned int >::Minimum = numeric_limits< unsigned int >::min();

    template< >
    const int KSNumerical< int >::Maximum = numeric_limits< int >::max();
    template< >
    const int KSNumerical< int >::Zero = 0;
    template< >
    const int KSNumerical< int >::Minimum = numeric_limits< int >::min();

    template< >
    const unsigned long KSNumerical< unsigned long >::Maximum = numeric_limits< unsigned long >::max();
    template< >
    const unsigned long KSNumerical< unsigned long >::Zero = 0;
    template< >
    const unsigned long KSNumerical< unsigned long >::Minimum = numeric_limits< unsigned long >::min();

    template< >
    const long KSNumerical< long >::Maximum = numeric_limits< long >::max();
    template< >
    const long KSNumerical< long >::Zero = 0;
    template< >
    const long KSNumerical< long >::Minimum = numeric_limits< long >::min();

    template< >
    const float KSNumerical< float >::Maximum = numeric_limits< float >::max();
    template< >
    const float KSNumerical< float >::Zero = 0.;
    template< >
    const float KSNumerical< float >::Minimum = numeric_limits< float >::min();

    template< >
    const double KSNumerical< double >::Maximum = numeric_limits< double >::max();
    template< >
    const double KSNumerical< double >::Zero = 0.;
    template< >
    const double KSNumerical< double >::Minimum = numeric_limits< double >::min();

    template< >
    const KTwoVector KSNumerical< KTwoVector >::Maximum = KTwoVector( KSNumerical< double >::Maximum, KSNumerical< double >::Maximum );
    template< >
    const KTwoVector KSNumerical< KTwoVector >::Zero = KTwoVector( KSNumerical< double >::Zero, KSNumerical< double >::Zero );
    template< >
    const KTwoVector KSNumerical< KTwoVector >::Minimum = KTwoVector( KSNumerical< double >::Minimum, KSNumerical< double >::Minimum );

    template< >
    const KThreeVector KSNumerical< KThreeVector >::Maximum = KThreeVector( KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum );
    template< >
    const KThreeVector KSNumerical< KThreeVector >::Zero = KThreeVector( KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero );
    template< >
    const KThreeVector KSNumerical< KThreeVector >::Minimum = KThreeVector( KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum );

    template< >
    const KTwoMatrix KSNumerical< KTwoMatrix >::Maximum = KTwoMatrix( KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum );
    template< >
    const KTwoMatrix KSNumerical< KTwoMatrix >::Zero = KTwoMatrix( KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero );
    template< >
    const KTwoMatrix KSNumerical< KTwoMatrix >::Minimum = KTwoMatrix( KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum );

    template< >
    const KThreeMatrix KSNumerical< KThreeMatrix >::Maximum = KThreeMatrix( KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum, KSNumerical< double >::Maximum );
    template< >
    const KThreeMatrix KSNumerical< KThreeMatrix >::Zero = KThreeMatrix( KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero, KSNumerical< double >::Zero );
    template< >
    const KThreeMatrix KSNumerical< KThreeMatrix >::Minimum = KThreeMatrix( KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum, KSNumerical< double >::Minimum );

}
