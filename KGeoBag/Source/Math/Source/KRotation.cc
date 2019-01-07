#include "KRotation.hh"

#include <cmath>
#include <cassert>

namespace KGeoBag
{

    KRotation::KRotation() :
        KThreeMatrix()
    {
        SetIdentity();
    }
    KRotation::KRotation( const KRotation& aRotation ) :
        KThreeMatrix( aRotation )
    {
    }
    KRotation::~KRotation()
    {
    }

    KRotation& KRotation::operator=(const KThreeMatrix& aMatrix)
    {
      fData[0] = aMatrix[0];
      fData[1] = aMatrix[1];
      fData[2] = aMatrix[2];

      fData[3] = aMatrix[3];
      fData[4] = aMatrix[4];
      fData[5] = aMatrix[5];

      fData[6] = aMatrix[6];
      fData[7] = aMatrix[7];
      fData[8] = aMatrix[8];

      return *this;
    }

    void KRotation::SetIdentity()
    {
        fData[0] = 1.;
        fData[1] = 0.;
        fData[2] = 0.;

        fData[3] = 0.;
        fData[4] = 1.;
        fData[5] = 0.;

        fData[6] = 0.;
        fData[7] = 0.;
        fData[8] = 1.;

        return;
    }

    void KRotation::SetAxisAngle( const KThreeVector& tAxis, const double& anAngle )
    {
        double tSine = sin( anAngle );
        double tCosine = cos( anAngle );
        KThreeVector tAxisUnit = tAxis.Unit();

        fData[0] = tCosine + (1 - tCosine) * tAxisUnit[0] * tAxisUnit[0];
        fData[1] = (1 - tCosine) * tAxisUnit[0] * tAxisUnit[1] - tSine * tAxisUnit[2];
        fData[2] = (1 - tCosine) * tAxisUnit[0] * tAxisUnit[2] + tSine * tAxisUnit[1];

        fData[3] = (1 - tCosine) * tAxisUnit[1] * tAxisUnit[0] + tSine * tAxisUnit[2];
        fData[4] = tCosine + (1 - tCosine) * tAxisUnit[1] * tAxisUnit[1];
        fData[5] = (1 - tCosine) * tAxisUnit[1] * tAxisUnit[2] - tSine * tAxisUnit[0];

        fData[6] = (1 - tCosine) * tAxisUnit[2] * tAxisUnit[0] - tSine * tAxisUnit[1];
        fData[7] = (1 - tCosine) * tAxisUnit[2] * tAxisUnit[1] + tSine * tAxisUnit[0];
        fData[8] = tCosine + (1 - tCosine) * tAxisUnit[2] * tAxisUnit[2];

        return;
    }
    void KRotation::SetAxisAngleInDegrees( const KThreeVector& tAxis, const double& anAngle )
    {
        SetAxisAngle( tAxis, KConst::Pi() / 180. * anAngle );
    }

    void KRotation::SetEulerAngles( const double& anAlpha, const double& aBeta, const double& aGamma )
    {
        double tSineAlpha = sin( anAlpha );
        double tCosineAlpha = cos( anAlpha );
        double tSineBeta = sin( aBeta );
        double tCosineBeta = cos( aBeta );
        double tSineGamma = sin( aGamma );
        double tCosineGamma = cos( aGamma );

        fData[0] = tCosineAlpha * tCosineGamma - tCosineBeta * tSineAlpha * tSineGamma;
        fData[1] = - tCosineAlpha * tSineGamma - tCosineBeta * tCosineGamma * tSineAlpha;
        fData[2] = tSineAlpha * tSineBeta;

        fData[3] = tCosineGamma * tSineAlpha + tCosineAlpha * tCosineBeta * tSineGamma;
        fData[4] = tCosineAlpha * tCosineBeta * tCosineGamma - tSineAlpha * tSineGamma;
        fData[5] = - tCosineAlpha * tSineBeta;

        fData[6] = tSineBeta * tSineGamma;
        fData[7] = tCosineGamma * tSineBeta;
        fData[8] = tCosineBeta;

        return;
    }
    void KRotation::SetEulerAnglesInDegrees( const double& anAlpha, const double& aBeta, const double& aGamma )
    {
        SetEulerAngles( KConst::Pi() / 180. * anAlpha, KConst::Pi() / 180. * aBeta, KConst::Pi() / 180. * aGamma );
    }

    void KRotation::SetEulerAngles( const vector<double>& anArray )
    {
        assert(anArray.size() == 3);
        SetEulerAngles( anArray[0], anArray[1], anArray[2] );
    }
    void KRotation::SetEulerAnglesInDegrees( const vector<double>& anArray )
    {
        assert(anArray.size() == 3);
        SetEulerAnglesInDegrees( anArray[0], anArray[1], anArray[2] );
    }

    void KRotation::SetEulerZYZAngles( const double& anAlpha, const double& aBeta, const double& aGamma )
    {
		// Euler Rotation in z,y',z'' convention
        double tSineAlpha = sin( anAlpha );
        double tCosineAlpha = cos( anAlpha );
        double tSineBeta = sin( aBeta );
        double tCosineBeta = cos( aBeta );
        double tSineGamma = sin( aGamma );
        double tCosineGamma = cos( aGamma );

        fData[0] = - tSineAlpha * tSineGamma + tCosineAlpha * tCosineBeta * tCosineGamma;
        fData[1] = - tSineAlpha * tCosineGamma - tCosineAlpha * tCosineBeta * tSineGamma;
        fData[2] = tCosineAlpha * tSineBeta;

        fData[3] = tCosineAlpha * tSineGamma + tSineAlpha * tCosineBeta * tCosineGamma;
        fData[4] = tCosineAlpha * tCosineGamma - tSineAlpha * tCosineBeta * tSineGamma;
        fData[5] = tSineAlpha * tSineBeta;

        fData[6] = - tSineBeta * tCosineGamma;
        fData[7] =  tSineBeta * tSineGamma;
        fData[8] = tCosineBeta;

        return;
    }
    void KRotation::SetEulerZYZAnglesInDegrees( const double& anAlpha, const double& aBeta, const double& aGamma )
    {
        SetEulerZYZAngles( KConst::Pi() / 180. * anAlpha, KConst::Pi() / 180. * aBeta, KConst::Pi() / 180. * aGamma );
    }

    void KRotation::SetEulerZYZAngles( const vector<double>& anArray )
    {
        assert(anArray.size() == 3);
        SetEulerZYZAngles( anArray[0], anArray[1], anArray[2] );
    }
    void KRotation::SetEulerZYZAnglesInDegrees( const vector<double>& anArray )
    {
        assert(anArray.size() == 3);
        SetEulerZYZAnglesInDegrees( anArray[0], anArray[1], anArray[2] );
    }

    void KRotation::SetRotatedFrame( const KThreeVector& x, const KThreeVector& y, const KThreeVector& z )
    {
        fData[0] = x[0];
        fData[1] = y[0];
        fData[2] = z[0];

        fData[3] = x[1];
        fData[4] = y[1];
        fData[5] = z[1];

        fData[6] = x[2];
        fData[7] = y[2];
        fData[8] = z[2];

        return;
    }

}
