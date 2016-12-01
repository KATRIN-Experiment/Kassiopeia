#ifndef Kassiopeia_KSGenConversion_h_
#define Kassiopeia_KSGenConversion_h_
/**
 @file
 @brief contains KSGenConversion
 @details
 <b>Revision History:</b>
 \verbatim
 Date       Name        Brief description
 -----------------------------------------------------
 Nov, 2011   wandkowsky     KASPER version

 \endverbatim
 */

/*!
 @class  Kassiopeia::KSGenConversion
 @author  wandkowsky

 @brief  KSGen conversion electrons handler

 @details
 <b>Detailed Description:</b><br>
 */

#include "KTextFile.h"

namespace Kassiopeia
{
    using std::vector;

    class KSGenConversion
    {

        public:
            KSGenConversion();
            ~KSGenConversion();

            bool Initialize( int isotope );

            void CreateCE( std::vector< int >& vacancy, std::vector< double >& energy );
            void SetForceCreation( bool asetting )
            {
                fForceCreation = asetting;
            }
            void SetIsotope( int isotope )
            {
                fIsotope = isotope;
            }

        protected:

            bool ReadData();
            katrin::KTextFile* fDataFile;

            bool fForceCreation;

            int fIsotope;
            int DoDoubleConversion;

            std::vector< std::vector< int > > fShell;
            std::vector< std::vector< int > > fDoubleConv;
            std::vector< std::vector< double > > fConvE;
            std::vector< std::vector< double > > fConvProb;
            std::vector< std::vector< double > > fConvProbNorm;

    };

} //namespace kassiopeia
#endif // KSGenConversion_H_
