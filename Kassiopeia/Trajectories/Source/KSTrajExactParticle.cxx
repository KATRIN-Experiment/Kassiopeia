#include "KSTrajExactParticle.h"
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

    KSMagneticField* KSTrajExactParticle::fMagneticFieldCalculator = NULL;
    KSElectricField* KSTrajExactParticle::fElectricFieldCalculator = NULL;
    double KSTrajExactParticle::fMass = 0.;
    double KSTrajExactParticle::fCharge = 0.;

    KSTrajExactParticle::KSTrajExactParticle() :
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
            fElectricPotential( 0. ),

            fGuidingCenter( 0., 0., 0. ),
            fLongMomentum( 0. ),
            fTransMomentum( 0. ),
            fLongVelocity( 0. ),
            fTransVelocity( 0. ),
            fCyclotronFrequency( 0. ),
            fOrbitalMagneticMoment( 0. ),

            fGetMagneticFieldPtr( &KSTrajExactParticle::RecalculateMagneticField ),
            fGetElectricFieldPtr( &KSTrajExactParticle::RecalculateElectricField ),
            fGetMagneticGradientPtr( &KSTrajExactParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialPtr( &KSTrajExactParticle::RecalculateElectricPotential )
    {
    }
    KSTrajExactParticle::~KSTrajExactParticle()
    {
    }

    //**********
    //assignment
    //**********

    void KSTrajExactParticle::PullFrom( const KSParticle& aParticle )
    {
        //trajmsg_debug( "exact particle pulling from particle:" << ret )

        if( fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator() )
        {
            //trajmsg_debug( "  magnetic calculator differs" << ret )
            fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

            fGetMagneticFieldPtr = &KSTrajExactParticle::RecalculateMagneticField;
            fGetMagneticGradientPtr = &KSTrajExactParticle::RecalculateMagneticGradient;
        }

        if( fElectricFieldCalculator != aParticle.GetElectricFieldCalculator() )
        {
            //trajmsg_debug( "  electric calculator differs" << ret )
            fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

            fGetElectricFieldPtr = &KSTrajExactParticle::RecalculateElectricField;
            fGetElectricPotentialPtr = &KSTrajExactParticle::RecalculateElectricPotential;
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

            fGetMagneticFieldPtr = &KSTrajExactParticle::RecalculateMagneticField;
            fGetElectricFieldPtr = &KSTrajExactParticle::RecalculateElectricField;
            fGetMagneticGradientPtr = &KSTrajExactParticle::RecalculateMagneticGradient;
            fGetElectricPotentialPtr = &KSTrajExactParticle::RecalculateElectricPotential;
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
    void KSTrajExactParticle::PushTo( KSParticle& aParticle ) const
    {
        //trajmsg_debug( "exact particle pushing to particle:" << eom )


        aParticle.SetLength( GetLength() );
        aParticle.SetPosition( GetPosition() );
        aParticle.SetMomentum( GetMomentum() );
        aParticle.SetTime( GetTime() );

        if( fGetMagneticFieldPtr == &KSTrajExactParticle::DoNothing )
        {
            aParticle.SetMagneticField( GetMagneticField() );
        }
        if( fGetElectricFieldPtr == &KSTrajExactParticle::DoNothing )
        {
            aParticle.SetElectricField( GetElectricField() );
        }
        if( fGetMagneticGradientPtr == &KSTrajExactParticle::DoNothing )
        {
            aParticle.SetMagneticGradient( GetMagneticGradient() );
        }
        if( fGetElectricPotentialPtr == &KSTrajExactParticle::DoNothing )
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

    void KSTrajExactParticle::SetMagneticFieldCalculator( KSMagneticField* anMagneticField )
    {
        fMagneticFieldCalculator = anMagneticField;
        return;
    }
    KSMagneticField* KSTrajExactParticle::GetMagneticFieldCalculator()
    {
        return fMagneticFieldCalculator;
    }

    void KSTrajExactParticle::SetElectricFieldCalculator( KSElectricField* anElectricField )
    {
        fElectricFieldCalculator = anElectricField;
        return;
    }
    KSElectricField* KSTrajExactParticle::GetElectricFieldCalculator()
    {
        return fElectricFieldCalculator;
    }

    //****************
    //static variables
    //****************

    void KSTrajExactParticle::SetMass( const double& aMass )
    {
        fMass = aMass;
        return;
    }
    const double& KSTrajExactParticle::GetMass()
    {
        return fMass;
    }

    void KSTrajExactParticle::SetCharge( const double& aCharge )
    {
        fCharge = aCharge;
        return;
    }
    const double& KSTrajExactParticle::GetCharge()
    {
        return fCharge;
    }

    //*****************
    //dynamic variables
    //*****************

    const double& KSTrajExactParticle::GetTime() const
    {
        fTime = fData[ 0 ];
        return fTime;
    }
    const double& KSTrajExactParticle::GetLength() const
    {
        fLength = fData[ 1 ];
        return fLength;
    }
    const KThreeVector& KSTrajExactParticle::GetPosition() const
    {
        fPosition.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPosition;
    }
    const KThreeVector& KSTrajExactParticle::GetMomentum() const
    {
        fMomentum.SetComponents( fData[ 5 ], fData[ 6 ], fData[ 7 ] );
        return fMomentum;
    }
    const KThreeVector& KSTrajExactParticle::GetVelocity() const
    {
        fVelocity = (1. / (GetMass() * GetLorentzFactor())) * GetMomentum();
        return fVelocity;
    }
    const double& KSTrajExactParticle::GetLorentzFactor() const
    {
        fLorentzFactor = sqrt( 1. + GetMomentum().MagnitudeSquared() / (GetMass() * GetMass() * KConst::C() * KConst::C()) );
        return fLorentzFactor;
    }
    const double& KSTrajExactParticle::GetKineticEnergy() const
    {
        fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
        return fKineticEnergy;
    }

    const KThreeVector& KSTrajExactParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldPtr)();
        return fMagneticField;
    }
    const KThreeVector& KSTrajExactParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldPtr)();
        return fElectricField;
    }
    const KThreeMatrix& KSTrajExactParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientPtr)();
        return fMagneticGradient;
    }
    const double& KSTrajExactParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialPtr)();
        return fElectricPotential;
    }

    const KThreeVector& KSTrajExactParticle::GetGuidingCenter() const
    {
        fGuidingCenter = GetPosition() + (1. / (GetCharge() * GetMagneticField().MagnitudeSquared())) * (GetMomentum().Cross( GetMagneticField() ));
        return fGuidingCenter;
    }
    const double& KSTrajExactParticle::GetLongMomentum() const
    {
        fLongMomentum = GetMomentum().Dot( GetMagneticField().Unit() );
        return fLongMomentum;
    }
    const double& KSTrajExactParticle::GetTransMomentum() const
    {
        fTransMomentum = (GetMomentum() - GetMomentum().Dot( GetMagneticField().Unit() ) * GetMagneticField().Unit()).Magnitude();
        return fTransMomentum;
    }
    const double& KSTrajExactParticle::GetLongVelocity() const
    {
        fLongVelocity = GetLongMomentum() / (GetMass() * GetLorentzFactor());
        return fLongVelocity;
    }
    const double& KSTrajExactParticle::GetTransVelocity() const
    {
        fTransVelocity = GetTransMomentum() / (GetMass() * GetLorentzFactor());
        return fTransVelocity;
    }
    const double& KSTrajExactParticle::GetCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( fCharge ) * GetMagneticField().Magnitude()) / (2. * KConst::Pi() * GetLorentzFactor() * GetMass());
        return fCyclotronFrequency;
    }
    const double& KSTrajExactParticle::GetOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = (GetTransMomentum() * GetTransMomentum()) / (2. * GetMagneticField().Magnitude() * GetMass());
        return fOrbitalMagneticMoment;
    }

    //*****
    //cache
    //*****

    void KSTrajExactParticle::DoNothing() const
    {
        return;
    }
    void KSTrajExactParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetPosition(), GetTime(), fMagneticField );
        fGetMagneticFieldPtr = &KSTrajExactParticle::DoNothing;
        return;
    }
    void KSTrajExactParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetPosition(), GetTime(), fElectricField );
        fGetElectricFieldPtr = &KSTrajExactParticle::DoNothing;
        return;
    }
    void KSTrajExactParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fMagneticGradient );
        fGetMagneticGradientPtr = &KSTrajExactParticle::DoNothing;
        return;
    }
    void KSTrajExactParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotential );
        fGetElectricPotentialPtr = &KSTrajExactParticle::DoNothing;
        return;
    }

}
