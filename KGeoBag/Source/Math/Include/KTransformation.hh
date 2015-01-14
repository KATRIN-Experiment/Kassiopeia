#ifndef KTRANSFORMATION_H_
#define KTRANSFORMATION_H_

#include "KThreeVector.hh"
#include "KRotation.hh"

namespace KGeoBag
{

    class KTransformation
    {
        public:
            KTransformation();
            KTransformation( const KTransformation& aTransformation );
            virtual ~KTransformation();

            void Apply( KThreeVector& point ) const;
            void ApplyRotation( KThreeVector& point ) const;
            void ApplyDisplacement( KThreeVector& point ) const;

            void ApplyInverse( KThreeVector& point ) const;
            void ApplyRotationInverse( KThreeVector& point ) const;
            void ApplyDisplacementInverse( KThreeVector& point ) const;

            //*****************
            //coordinate system
            //*****************

        public:
            void SetOrigin( const KThreeVector& origin );
            void SetFrameAxisAngle( const double& angle, const double& theta, const double& phi );
            void SetFrameEuler( const double& alpha, const double& beta, const double& gamma );
            void SetXAxis( const KThreeVector& xaxis );
            void SetYAxis( const KThreeVector& yaxis );
            void SetZAxis( const KThreeVector& zaxis );
            const KThreeVector& GetOrigin() const;
            const KThreeVector& GetXAxis() const;
            const KThreeVector& GetYAxis() const;
            const KThreeVector& GetZAxis() const;

        private:
            void LocalFromGlobal( const KThreeVector& point, KThreeVector& target ) const;
            void GlobalFromLocal( const KThreeVector& point, KThreeVector& target ) const;

            KThreeVector fOrigin;
            KThreeVector fXAxis;
            KThreeVector fYAxis;
            KThreeVector fZAxis;

            //********
            //rotation
            //********

        public:
            void SetRotationAxisAngle( const double& angle, const double& theta, const double& phi );
            void SetRotationEuler( const double& phi, const double& theta, const double& psi );
            void SetRotationZYZEuler( const double& phi, const double& theta, const double& psi );
            void SetRotatedFrame( const KThreeVector& x, const KThreeVector& y, const KThreeVector& z );
            const KRotation& GetRotation() const;

        private:
            KRotation fRotation;
            KRotation fRotationInverse;

            //************
            //displacement
            //************

        public:
            void SetDisplacement( const double& xdisp, const double& ydisp, const double& zdisp );
            void SetDisplacement( const KThreeVector& disp );
            const KThreeVector& GetDisplacement() const;

        private:
            KThreeVector fDisplacement;
    };

    inline const KThreeVector& KTransformation::GetOrigin() const
    {
        return fOrigin;
    }
    inline const KThreeVector& KTransformation::GetXAxis() const
    {
        return fXAxis;
    }
    inline const KThreeVector& KTransformation::GetYAxis() const
    {
        return fYAxis;
    }
    inline const KThreeVector& KTransformation::GetZAxis() const
    {
        return fZAxis;
    }

    inline const KThreeVector& KTransformation::GetDisplacement() const
    {
        return fDisplacement;
    }

    inline const KRotation& KTransformation::GetRotation() const
    {
        return fRotation;
    }

}

#endif
