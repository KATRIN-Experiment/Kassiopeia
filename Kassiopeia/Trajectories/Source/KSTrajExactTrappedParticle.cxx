#include "KSTrajExactTrappedParticle.h"
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

    KSMagneticField* KSTrajExactTrappedParticle::fMagneticFieldCalculator = NULL;
    KSElectricField* KSTrajExactTrappedParticle::fElectricFieldCalculator = NULL;
    double KSTrajExactTrappedParticle::fMass = 0.;
    double KSTrajExactTrappedParticle::fCharge = 0.;

    KSTrajExactTrappedParticle::KSTrajExactTrappedParticle() :
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

            fGetMagneticFieldPtr( &KSTrajExactTrappedParticle::RecalculateMagneticField ),
            fGetElectricFieldPtr( &KSTrajExactTrappedParticle::RecalculateElectricField ),
            fGetMagneticGradientPtr( &KSTrajExactTrappedParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialPtr( &KSTrajExactTrappedParticle::RecalculateElectricPotential ),
            fGetElectricGradientPtr( &KSTrajExactTrappedParticle::RecalculateElectricGradient )
    {
    }
    KSTrajExactTrappedParticle::~KSTrajExactTrappedParticle()
    {
    }

    //**********
    //assignment
    //**********

    void KSTrajExactTrappedParticle::PullFrom( const KSParticle& aParticle )
    {
        //trajmsg_debug( "ExactTrapped particle pulling from particle:" << ret )

        if( fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator() )
        {
            //trajmsg_debug( "  magnetic calculator differs" << ret )
            fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

            fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::RecalculateMagneticField;
            fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::RecalculateMagneticGradient;
        }

        if( fElectricFieldCalculator != aParticle.GetElectricFieldCalculator() )
        {
            //trajmsg_debug( "  electric calculator differs" << ret )
            fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

            fGetElectricFieldPtr = &KSTrajExactTrappedParticle::RecalculateElectricField;
            fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::RecalculateElectricPotential;
            fGetElectricGradientPtr = &KSTrajExactTrappedParticle::RecalculateElectricGradient;
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

            fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::RecalculateMagneticField;
            fGetElectricFieldPtr = &KSTrajExactTrappedParticle::RecalculateElectricField;
            fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::RecalculateMagneticGradient;
            fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::RecalculateElectricPotential;
            fGetElectricGradientPtr = &KSTrajExactTrappedParticle::RecalculateElectricGradient;
        }

        if( GetMomentum() != aParticle.GetMomentum() )
        {
            //trajmsg_debug( "  momentum differs" << ret )
            fMomentum = aParticle.GetMomentum();

            fData[ 5 ] = fMomentum.X();
            fData[ 6 ] = fMomentum.Y();
            fData[ 7 ] = fMomentum.Z();
        }

        //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
        //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
        //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
        //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

        return;
    }
    void KSTrajExactTrappedParticle::PushTo( KSParticle& aParticle ) const
    {
        //trajmsg_debug( "ExactTrapped particle pushing to particle:" << eom )


        aParticle.SetLength( GetLength() );
        aParticle.SetPosition( GetPosition() );
        aParticle.SetMomentum( GetMomentum() );
        aParticle.SetTime( GetTime() );

        if( fGetMagneticFieldPtr == &KSTrajExactTrappedParticle::DoNothing )
        {
            aParticle.SetMagneticField( GetMagneticField() );
        }
        if( fGetElectricFieldPtr == &KSTrajExactTrappedParticle::DoNothing )
        {
            aParticle.SetElectricField( GetElectricField() );
        }
        if( fGetMagneticGradientPtr == &KSTrajExactTrappedParticle::DoNothing )
        {
            aParticle.SetMagneticGradient( GetMagneticGradient() );
        }
        if( fGetElectricPotentialPtr == &KSTrajExactTrappedParticle::DoNothing )
        {
            aParticle.SetElectricPotential( GetElectricPotential() );
        }

        //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
        //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
        //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
        //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

        return;
    }

    //***********
    //calculators
    //***********

    void KSTrajExactTrappedParticle::SetMagneticFieldCalculator( KSMagneticField* anMagneticField )
    {
        fMagneticFieldCalculator = anMagneticField;
        return;
    }
    KSMagneticField* KSTrajExactTrappedParticle::GetMagneticFieldCalculator()
    {
        return fMagneticFieldCalculator;
    }

    void KSTrajExactTrappedParticle::SetElectricFieldCalculator( KSElectricField* anElectricField )
    {
        fElectricFieldCalculator = anElectricField;
        return;
    }
    KSElectricField* KSTrajExactTrappedParticle::GetElectricFieldCalculator()
    {
        return fElectricFieldCalculator;
    }

    //****************
    //static variables
    //****************

    void KSTrajExactTrappedParticle::SetMass( const double& aMass )
    {
        fMass = aMass;
        return;
    }
    const double& KSTrajExactTrappedParticle::GetMass()
    {
        return fMass;
    }

    void KSTrajExactTrappedParticle::SetCharge( const double& aCharge )
    {
        fCharge = aCharge;
        return;
    }
    const double& KSTrajExactTrappedParticle::GetCharge()
    {
        return fCharge;
    }

    //*****************
    //dynamic variables
    //*****************

    const double& KSTrajExactTrappedParticle::GetTime() const
    {
        fTime = fData[ 0 ];
        return fTime;
    }
    const double& KSTrajExactTrappedParticle::GetLength() const
    {
        fLength = fData[ 1 ];
        return fLength;
    }
    const KThreeVector& KSTrajExactTrappedParticle::GetPosition() const
    {
        fPosition.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPosition;
    }
    const KThreeVector& KSTrajExactTrappedParticle::GetMomentum() const
    {
        fMomentum.SetComponents( fData[ 5 ], fData[ 6 ], fData[ 7 ] );
        return fMomentum;
    }
    const KThreeVector& KSTrajExactTrappedParticle::GetVelocity() const
    {
        fVelocity = (1. / (GetMass() * GetLorentzFactor())) * GetMomentum();
        return fVelocity;
    }
    const double& KSTrajExactTrappedParticle::GetLorentzFactor() const
    {
        fLorentzFactor = sqrt( 1. + GetMomentum().MagnitudeSquared() / (GetMass() * GetMass() * KConst::C() * KConst::C()) );
        return fLorentzFactor;
    }
    const double& KSTrajExactTrappedParticle::GetKineticEnergy() const
    {
        fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
        return fKineticEnergy;
    }

    const KThreeVector& KSTrajExactTrappedParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldPtr)();
        return fMagneticField;
    }
    const KThreeVector& KSTrajExactTrappedParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldPtr)();
        return fElectricField;
    }
    const KThreeMatrix& KSTrajExactTrappedParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientPtr)();
        return fMagneticGradient;
    }
    const KThreeMatrix& KSTrajExactTrappedParticle::GetElectricGradient() const
    {
        (this->*fGetElectricGradientPtr)();
        return fElectricGradient;
    }
    const double& KSTrajExactTrappedParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialPtr)();
        return fElectricPotential;
    }

    const KThreeVector& KSTrajExactTrappedParticle::GetGuidingCenter() const
    {
        fGuidingCenter = GetPosition() + (1. / (GetCharge() * GetMagneticField().MagnitudeSquared())) * (GetMomentum().Cross( GetMagneticField() ));
        return fGuidingCenter;
    }
    const double& KSTrajExactTrappedParticle::GetLongMomentum() const
    {
        fLongMomentum = GetMomentum().Dot( GetMagneticField().Unit() );
        return fLongMomentum;
    }
    const double& KSTrajExactTrappedParticle::GetTransMomentum() const
    {
        fTransMomentum = (GetMomentum() - GetMomentum().Dot( GetMagneticField().Unit() ) * GetMagneticField().Unit()).Magnitude();
        return fTransMomentum;
    }
    const double& KSTrajExactTrappedParticle::GetLongVelocity() const
    {
        fLongVelocity = GetLongMomentum() / (GetMass() * GetLorentzFactor());
        return fLongVelocity;
    }
    const double& KSTrajExactTrappedParticle::GetTransVelocity() const
    {
        fTransVelocity = GetTransMomentum() / (GetMass() * GetLorentzFactor());
        return fTransVelocity;
    }
    const double& KSTrajExactTrappedParticle::GetCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( fCharge ) * GetMagneticField().Magnitude()) / (2. * KConst::Pi() * GetLorentzFactor() * GetMass());
        return fCyclotronFrequency;
    }
    const double& KSTrajExactTrappedParticle::GetOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = (GetTransMomentum() * GetTransMomentum()) / (2. * GetMagneticField().Magnitude() * GetMass());
        return fOrbitalMagneticMoment;
    }

    //*****
    //cache
    //*****

    void KSTrajExactTrappedParticle::DoNothing() const
    {
        return;
    }
    void KSTrajExactTrappedParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetPosition(), GetTime(), fMagneticField );
        fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::DoNothing;
        return;
    }
    void KSTrajExactTrappedParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetPosition(), GetTime(), fElectricField );
        fGetElectricFieldPtr = &KSTrajExactTrappedParticle::DoNothing;
        return;
    }
    void KSTrajExactTrappedParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fMagneticGradient );
        fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::DoNothing;
        return;
    }
    void KSTrajExactTrappedParticle::RecalculateElectricGradient() const
    {
        fElectricFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fElectricGradient );
        fGetElectricGradientPtr = &KSTrajExactTrappedParticle::DoNothing;
        return;
    }
    void KSTrajExactTrappedParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotential );
        fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::DoNothing;
        return;
    }

}
