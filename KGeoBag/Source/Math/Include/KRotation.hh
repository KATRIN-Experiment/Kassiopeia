#ifndef KROTATION_H_
#define KROTATION_H_

#include "KThreeMatrix.hh"

namespace KGeoBag
{

    class KRotation :
        public KThreeMatrix
    {
        public:
            KRotation();
            KRotation( const KRotation& aRotation );
            virtual ~KRotation();

            KRotation& operator=( const KThreeMatrix& aMatrix );

            void SetIdentity();
            void SetAxisAngle( const KThreeVector& anAxis, const double& anAngle );
            void SetEulerAngles( const double& anAlpha, const double& aBeta, const double& aGamma );
            void SetEulerZYZAngles( const double& anAlpha, const double& aBeta, const double& aGamma );
            void SetRotatedFrame( const KThreeVector& x, const KThreeVector& y, const KThreeVector& z );
    };

}

#endif
