#include "KSTrajAdiabaticSpinParticle.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"
using katrin::KConst;

#include <cmath>

namespace Kassiopeia
{

    //0 is time
    //1 is length
    //2 is x component of position
    //3 is y component of position
    //4 is z component of position
    //5 is x component of momentum
    //6 is y component of momentum
    //7 is z component of momentum
    //8 is B-aligned component of spin
    //9 is B-perp angle of spin

    KSMagneticField* KSTrajAdiabaticSpinParticle::fMagneticFieldCalculator = NULL;
    KSElectricField* KSTrajAdiabaticSpinParticle::fElectricFieldCalculator = NULL;
    double KSTrajAdiabaticSpinParticle::fMass = 0.;
    double KSTrajAdiabaticSpinParticle::fCharge = 0.;
    double KSTrajAdiabaticSpinParticle::fSpinMagnitude = 0.;
    double KSTrajAdiabaticSpinParticle::fGyromagneticRatio = 0.;

    KSTrajAdiabaticSpinParticle::KSTrajAdiabaticSpinParticle() :
            fTime( 0. ),
            fLength( 0. ),
            fPosition( 0., 0., 0. ),
            fMomentum( 0., 0., 0. ),
            fVelocity( 0., 0., 0. ),
            fLorentzFactor( 0. ),
            fKineticEnergy( 0. ),

            fMagneticField( 0., 0., 0. ),
            fElectricField( 0., 0., 0. ),
            fMagneticGradient( 0., 0., 0., 0., 0., 0., 0., 0., 0. ),
            fElectricGradient( 0., 0., 0., 0., 0., 0., 0., 0., 0. ),
            fElectricPotential( 0. ),

            fGuidingCenter( 0., 0., 0. ),
            fLongMomentum( 0. ),
            fTransMomentum( 0. ),
            fLongVelocity( 0. ),
            fTransVelocity( 0. ),
            fCyclotronFrequency( 0. ),
            fOrbitalMagneticMoment( 0. ),

            fAlignedSpin( 0. ),
            fSpinAngle( 0. ),

            fGetMagneticFieldPtr( &KSTrajAdiabaticSpinParticle::RecalculateMagneticField ),
            fGetElectricFieldPtr( &KSTrajAdiabaticSpinParticle::RecalculateElectricField ),
            fGetMagneticGradientPtr( &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialPtr( &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential ),
            fGetElectricGradientPtr( &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient )
    {
    }
    KSTrajAdiabaticSpinParticle::~KSTrajAdiabaticSpinParticle()
    {
    }

    //**********
    //assignment
    //**********

    void KSTrajAdiabaticSpinParticle::PullFrom( const KSParticle& aParticle )
    {
        //trajmsg_debug( "AdiabaticSpin particle pulling from particle:" << ret )

        if( fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator() )
        {
            //trajmsg_debug( "  magnetic calculator differs" << ret )
            fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

            fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticField;
            fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient;
        }

        if( fElectricFieldCalculator != aParticle.GetElectricFieldCalculator() )
        {
            //trajmsg_debug( "  electric calculator differs" << ret )
            fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

            fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricField;
            fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential;
            fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient;
        }

        if( GetMass() != aParticle.GetMass() )
        {
            //trajmsg_debug( "  mass differs" << ret )
            fMass = aParticle.GetMass();
        }

        if( GetCharge() != aParticle.GetCharge() )
        {
            //trajmsg_debug( "  charge differs" << ret )
            fCharge = aParticle.GetCharge();
        }

        if( GetSpinMagnitude()  != aParticle.GetSpinMagnitude() )
        {
            //trajmsg_debug( "  charge differs" << ret )
            fSpinMagnitude = aParticle.GetSpinMagnitude();
        }
        if( GetGyromagneticRatio()  != aParticle.GetGyromagneticRatio() )
        {
            //trajmsg_debug( "  charge differs" << ret )
            fGyromagneticRatio = aParticle.GetGyromagneticRatio();
        }

        if( GetTime() != aParticle.GetTime() || GetPosition() != aParticle.GetPosition() )
        {
            //trajmsg_debug( "  time or position differs" << ret )

            fTime = aParticle.GetTime();
            fLength = aParticle.GetLength();
            fPosition = aParticle.GetPosition();

            fData[ 0 ] = fTime;
            fData[ 1 ] = fLength;
            fData[ 2 ] = fPosition.X();
            fData[ 3 ] = fPosition.Y();
            fData[ 4 ] = fPosition.Z();

            fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticField;
            fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricField;
            fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient;
            fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential;
            fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient;
        }

        if( GetMomentum() != aParticle.GetMomentum() )
        {
            //trajmsg_debug( "  momentum differs" << ret )
            fMomentum = aParticle.GetMomentum();

            fData[ 5 ] = fMomentum.X();
            fData[ 6 ] = fMomentum.Y();
            fData[ 7 ] = fMomentum.Z();
        }

        aParticle.RecalculateMagneticField();
        aParticle.RecalculateSpinGlobal();

        if( GetAlignedSpin() != aParticle.GetAlignedSpin() )
        {
            fAlignedSpin = aParticle.GetAlignedSpin();

            fData[ 8 ] = fAlignedSpin;
        }

        if( GetSpinAngle() != aParticle.GetSpinAngle() )
        {
            fSpinAngle = aParticle.GetSpinAngle();

            fData[ 9 ] = std::fmod( fSpinAngle, 2.*KConst::Pi() );
        }

        FixSpin();

        //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
        //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
        //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
        //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

        return;
    }
    void KSTrajAdiabaticSpinParticle::PushTo( KSParticle& aParticle ) const
    {
        //trajmsg_debug( "AdiabaticSpin particle pushing to particle:" << eom )


        aParticle.SetLength( GetLength() );
        aParticle.SetPosition( GetPosition() );
        aParticle.SetMomentum( GetMomentum() );
        aParticle.SetTime( GetTime() );

        aParticle.SetAlignedSpin( GetAlignedSpin() );
        aParticle.SetSpinAngle( std::fmod( GetSpinAngle(), 2.*KConst::Pi() ) );

        if( fGetMagneticFieldPtr == &KSTrajAdiabaticSpinParticle::DoNothing )
        {
            aParticle.SetMagneticField( GetMagneticField() );
        }
        if( fGetElectricFieldPtr == &KSTrajAdiabaticSpinParticle::DoNothing )
        {
            aParticle.SetElectricField( GetElectricField() );
        }
        if( fGetMagneticGradientPtr == &KSTrajAdiabaticSpinParticle::DoNothing )
        {
            aParticle.SetMagneticGradient( GetMagneticGradient() );
        }
        if( fGetElectricPotentialPtr == &KSTrajAdiabaticSpinParticle::DoNothing )
        {
            aParticle.SetElectricPotential( GetElectricPotential() );
        }

        aParticle.RecalculateSpinGlobal();

        //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
        //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
        //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
        //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

        return;
    }

    //***********
    //calculators
    //***********

    void KSTrajAdiabaticSpinParticle::SetMagneticFieldCalculator( KSMagneticField* anMagneticField )
    {
        fMagneticFieldCalculator = anMagneticField;
        return;
    }
    KSMagneticField* KSTrajAdiabaticSpinParticle::GetMagneticFieldCalculator()
    {
        return fMagneticFieldCalculator;
    }

    void KSTrajAdiabaticSpinParticle::SetElectricFieldCalculator( KSElectricField* anElectricField )
    {
        fElectricFieldCalculator = anElectricField;
        return;
    }
    KSElectricField* KSTrajAdiabaticSpinParticle::GetElectricFieldCalculator()
    {
        return fElectricFieldCalculator;
    }

    //****************
    //static variables
    //****************

    void KSTrajAdiabaticSpinParticle::SetMass( const double& aMass )
    {
        fMass = aMass;
        return;
    }
    const double& KSTrajAdiabaticSpinParticle::GetMass()
    {
        return fMass;
    }

    void KSTrajAdiabaticSpinParticle::SetCharge( const double& aCharge )
    {
        fCharge = aCharge;
        return;
    }
    const double& KSTrajAdiabaticSpinParticle::GetCharge()
    {
        return fCharge;
    }

    void KSTrajAdiabaticSpinParticle::SetSpinMagnitude( const double& aSpinMagnitude )
    {
        fSpinMagnitude = aSpinMagnitude;
        return;
    }
    const double& KSTrajAdiabaticSpinParticle::GetSpinMagnitude()
    {
        return fSpinMagnitude;
    }

    void KSTrajAdiabaticSpinParticle::SetGyromagneticRatio( const double& aGyromagneticRatio )
    {
        fGyromagneticRatio = aGyromagneticRatio;
        return;
    }
    const double& KSTrajAdiabaticSpinParticle::GetGyromagneticRatio()
    {
        return fGyromagneticRatio;
    }

    void KSTrajAdiabaticSpinParticle::FixSpin()
    {
        if ( fAlignedSpin > 0.99999 )
        {
            fData[ 8 ] = 0.99999;
            fAlignedSpin = 0.99999;
        }
        if ( fAlignedSpin < -0.99999 )
        {
            fData[ 8 ] = -0.99999;
            fAlignedSpin = -0.99999;
        }
    }

    //*****************
    //dynamic variables
    //*****************

    const double& KSTrajAdiabaticSpinParticle::GetTime() const
    {
        fTime = fData[ 0 ];
        return fTime;
    }
    const double& KSTrajAdiabaticSpinParticle::GetLength() const
    {
        fLength = fData[ 1 ];
        return fLength;
    }
    const KThreeVector& KSTrajAdiabaticSpinParticle::GetPosition() const
    {
        fPosition.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPosition;
    }
    const KThreeVector& KSTrajAdiabaticSpinParticle::GetMomentum() const
    {
        fMomentum.SetComponents( fData[ 5 ], fData[ 6 ], fData[ 7 ] );
        return fMomentum;
    }
    const KThreeVector& KSTrajAdiabaticSpinParticle::GetVelocity() const
    {
        fVelocity = (1. / (GetMass() * GetLorentzFactor())) * GetMomentum();
        return fVelocity;
    }
    const double& KSTrajAdiabaticSpinParticle::GetLorentzFactor() const
    {
        fLorentzFactor = sqrt( 1. + GetMomentum().MagnitudeSquared() / (GetMass() * GetMass() * KConst::C() * KConst::C()) );
        return fLorentzFactor;
    }
    const double& KSTrajAdiabaticSpinParticle::GetKineticEnergy() const
    {
        fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
        return fKineticEnergy;
    }

    const KThreeVector& KSTrajAdiabaticSpinParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldPtr)();
        return fMagneticField;
    }
    const KThreeVector& KSTrajAdiabaticSpinParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldPtr)();
        return fElectricField;
    }
    const KThreeMatrix& KSTrajAdiabaticSpinParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientPtr)();
        return fMagneticGradient;
    }
    const KThreeMatrix& KSTrajAdiabaticSpinParticle::GetElectricGradient() const
    {
        (this->*fGetElectricGradientPtr)();
        return fElectricGradient;
    }
    const double& KSTrajAdiabaticSpinParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialPtr)();
        return fElectricPotential;
    }

    const KThreeVector& KSTrajAdiabaticSpinParticle::GetGuidingCenter() const
    {
        fGuidingCenter = GetPosition() + (1. / (GetCharge() * GetMagneticField().MagnitudeSquared())) * (GetMomentum().Cross( GetMagneticField() ));
        return fGuidingCenter;
    }
    const double& KSTrajAdiabaticSpinParticle::GetLongMomentum() const
    {
        fLongMomentum = GetMomentum().Dot( GetMagneticField().Unit() );
        return fLongMomentum;
    }
    const double& KSTrajAdiabaticSpinParticle::GetTransMomentum() const
    {
        fTransMomentum = (GetMomentum() - GetMomentum().Dot( GetMagneticField().Unit() ) * GetMagneticField().Unit()).Magnitude();
        return fTransMomentum;
    }
    const double& KSTrajAdiabaticSpinParticle::GetLongVelocity() const
    {
        fLongVelocity = GetLongMomentum() / (GetMass() * GetLorentzFactor());
        return fLongVelocity;
    }
    const double& KSTrajAdiabaticSpinParticle::GetTransVelocity() const
    {
        fTransVelocity = GetTransMomentum() / (GetMass() * GetLorentzFactor());
        return fTransVelocity;
    }
    const double& KSTrajAdiabaticSpinParticle::GetCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( fCharge ) * GetMagneticField().Magnitude()) / (2. * KConst::Pi() * GetLorentzFactor() * GetMass());
        return fCyclotronFrequency;
    }
    const double& KSTrajAdiabaticSpinParticle::GetSpinPrecessionFrequency() const
    {
        fSpinPrecessionFrequency = std::abs( GetGyromagneticRatio() * GetMagneticField().Magnitude() );
        return fSpinPrecessionFrequency;
    }
    const double& KSTrajAdiabaticSpinParticle::GetOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = (GetTransMomentum() * GetTransMomentum()) / (2. * GetMagneticField().Magnitude() * GetMass());
        return fOrbitalMagneticMoment;
    }

    const double& KSTrajAdiabaticSpinParticle::GetAlignedSpin() const
    {
        //FixSpin();
        fAlignedSpin = fData[ 8 ];
        return fAlignedSpin;
    }
    const double& KSTrajAdiabaticSpinParticle::GetSpinAngle() const
    {
        //FixSpin();
        fSpinAngle = fData[ 9 ];
        return fSpinAngle;
    }

    //*****
    //cache
    //*****

    void KSTrajAdiabaticSpinParticle::DoNothing() const
    {
        return;
    }
    void KSTrajAdiabaticSpinParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetPosition(), GetTime(), fMagneticField );
        fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticSpinParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetPosition(), GetTime(), fElectricField );
        fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fMagneticGradient );
        fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticSpinParticle::RecalculateElectricGradient() const
    {
        fElectricFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fElectricGradient );
        fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticSpinParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotential );
        fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::DoNothing;
        return;
    }

}
