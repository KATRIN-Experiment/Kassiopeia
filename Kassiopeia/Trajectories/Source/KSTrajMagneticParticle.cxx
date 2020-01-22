#include "KSTrajMagneticParticle.h"
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

    KSMagneticField* KSTrajMagneticParticle::fMagneticFieldCalculator = NULL;
    KSElectricField* KSTrajMagneticParticle::fElectricFieldCalculator = NULL;
    double KSTrajMagneticParticle::fMass = 0.;
    double KSTrajMagneticParticle::fCharge = 0.;

    KSTrajMagneticParticle::KSTrajMagneticParticle() :
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

            fGetMagneticFieldPtr( &KSTrajMagneticParticle::RecalculateMagneticField ),
            fGetElectricFieldPtr( &KSTrajMagneticParticle::RecalculateElectricField ),
            fGetMagneticGradientPtr( &KSTrajMagneticParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialPtr( &KSTrajMagneticParticle::RecalculateElectricPotential )
    {
    }
    KSTrajMagneticParticle::KSTrajMagneticParticle( const KSTrajMagneticParticle& aParticle ) :
            KSMathArray<5>( aParticle ),
            fTime( aParticle.fTime ),
            fLength( aParticle.fLength ),
            fPosition( aParticle.fPosition ),
            fMomentum( aParticle.fMomentum ),
            fVelocity( aParticle.fVelocity ),
            fLorentzFactor( aParticle.fLorentzFactor ),
            fKineticEnergy( aParticle.fKineticEnergy ),

            fMagneticField( aParticle.fMagneticField ),
            fElectricField( aParticle.fElectricField ),
            fMagneticGradient( aParticle.fMagneticGradient ),
            fElectricPotential( aParticle.fElectricPotential ),

            fGuidingCenter( aParticle.fGuidingCenter ),
            fLongMomentum( aParticle.fLongMomentum ),
            fTransMomentum( aParticle.fTransMomentum ),
            fLongVelocity( aParticle.fLongVelocity ),
            fTransVelocity( aParticle.fTransVelocity ),
            fCyclotronFrequency( aParticle.fCyclotronFrequency ),
            fOrbitalMagneticMoment( aParticle.fOrbitalMagneticMoment ),

            fGetMagneticFieldPtr( aParticle.fGetMagneticFieldPtr ),
            fGetElectricFieldPtr( aParticle.fGetElectricFieldPtr ),
            fGetMagneticGradientPtr( aParticle.fGetMagneticGradientPtr ),
            fGetElectricPotentialPtr( aParticle.fGetElectricPotentialPtr )
    {
    }
    KSTrajMagneticParticle::~KSTrajMagneticParticle()
    {
    }

    //**********
    //assignment
    //**********

    void KSTrajMagneticParticle::PullFrom( const KSParticle& aParticle )
    {
        //trajmsg_debug( "magnetic particle pulling from particle:" << ret )

        if( fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator() )
        {
            //trajmsg_debug( "  magnetic calculator differs" << ret )
            fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

            fGetMagneticFieldPtr = &KSTrajMagneticParticle::RecalculateMagneticField;
            fGetMagneticGradientPtr = &KSTrajMagneticParticle::RecalculateMagneticGradient;
        }

        if( fElectricFieldCalculator != aParticle.GetElectricFieldCalculator() )
        {
            //trajmsg_debug( "  electric calculator differs" << ret )
            fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

            fGetElectricFieldPtr = &KSTrajMagneticParticle::RecalculateElectricField;
            fGetElectricPotentialPtr = &KSTrajMagneticParticle::RecalculateElectricPotential;
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

            fGetMagneticFieldPtr = &KSTrajMagneticParticle::RecalculateMagneticField;
            fGetElectricFieldPtr = &KSTrajMagneticParticle::RecalculateElectricField;
            fGetMagneticGradientPtr = &KSTrajMagneticParticle::RecalculateMagneticGradient;
            fGetElectricPotentialPtr = &KSTrajMagneticParticle::RecalculateElectricPotential;
        }

        //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
        //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
        //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
        //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

        return;
    }
    void KSTrajMagneticParticle::PushTo( KSParticle& aParticle )
    {
        //trajmsg_debug( "magnetic particle pushing to particle:" << eom )

        aParticle.SetTime( GetTime() );
        aParticle.SetLength( GetLength() );
        aParticle.SetPosition( GetPosition() );
        aParticle.SetMomentum( GetMomentum() );

        if( fGetMagneticFieldPtr == &KSTrajMagneticParticle::DoNothing )
        {
            aParticle.SetMagneticField( GetMagneticField() );
        }
        if( fGetElectricFieldPtr == &KSTrajMagneticParticle::DoNothing )
        {
            aParticle.SetElectricField( GetElectricField() );
        }
        if( fGetMagneticGradientPtr == &KSTrajMagneticParticle::DoNothing )
        {
            aParticle.SetMagneticGradient( GetMagneticGradient() );
        }
        if( fGetElectricPotentialPtr == &KSTrajMagneticParticle::DoNothing )
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

    void KSTrajMagneticParticle::SetMagneticFieldCalculator( KSMagneticField* anMagneticField )
    {
        fMagneticFieldCalculator = anMagneticField;
        return;
    }
    KSMagneticField* KSTrajMagneticParticle::GetMagneticFieldCalculator()
    {
        return fMagneticFieldCalculator;
    }

    void KSTrajMagneticParticle::SetElectricFieldCalculator( KSElectricField* anElectricField )
    {
        fElectricFieldCalculator = anElectricField;
        return;
    }
    KSElectricField* KSTrajMagneticParticle::GetElectricFieldCalculator()
    {
        return fElectricFieldCalculator;
    }

    //****************
    //static variables
    //****************

    void KSTrajMagneticParticle::SetMass( const double& aMass )
    {
        fMass = aMass;
        return;
    }
    const double& KSTrajMagneticParticle::GetMass()
    {
        return fMass;
    }

    void KSTrajMagneticParticle::SetCharge( const double& aCharge )
    {
        fCharge = aCharge;
        return;
    }
    const double& KSTrajMagneticParticle::GetCharge()
    {
        return fCharge;
    }

    //*****************
    //dynamic variables
    //*****************

    const double& KSTrajMagneticParticle::GetTime() const
    {
        fTime = fData[ 0 ];
        return fTime;
    }
    const double& KSTrajMagneticParticle::GetLength() const
    {
        fLength = fData[ 1 ];
        return fLength;
    }
    const KThreeVector& KSTrajMagneticParticle::GetPosition() const
    {
        fPosition.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPosition;
    }
    const KThreeVector& KSTrajMagneticParticle::GetMomentum() const
    {
        fMomentum = GetMass() * GetVelocity() * GetLorentzFactor();
        return fMomentum;
    }
    const KThreeVector& KSTrajMagneticParticle::GetVelocity() const
    {
        fVelocity = GetMagneticField().Unit();
        return fVelocity;
    }
    const double& KSTrajMagneticParticle::GetLorentzFactor() const
    {
        fLorentzFactor = 1. / sqrt( 1. - GetVelocity().MagnitudeSquared() / (KConst::C() * KConst::C()) );
        return fLorentzFactor;
    }
    const double& KSTrajMagneticParticle::GetKineticEnergy() const
    {
        fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
        return fKineticEnergy;
    }

    const KThreeVector& KSTrajMagneticParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldPtr)();
        return fMagneticField;
    }
    const KThreeVector& KSTrajMagneticParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldPtr)();
        return fElectricField;
    }
    const KThreeMatrix& KSTrajMagneticParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientPtr)();
        return fMagneticGradient;
    }
    const double& KSTrajMagneticParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialPtr)();
        return fElectricPotential;
    }

    const KThreeVector& KSTrajMagneticParticle::GetGuidingCenter() const
    {
        fGuidingCenter = GetPosition();
        return fGuidingCenter;
    }
    const double& KSTrajMagneticParticle::GetLongMomentum() const
    {
        fLongMomentum = GetMomentum().Magnitude();
        return fLongMomentum;
    }
    const double& KSTrajMagneticParticle::GetTransMomentum() const
    {
        fTransMomentum = 0.;
        return fTransMomentum;
    }
    const double& KSTrajMagneticParticle::GetLongVelocity() const
    {
        fLongVelocity = GetVelocity().Magnitude();
        return fLongVelocity;
    }
    const double& KSTrajMagneticParticle::GetTransVelocity() const
    {
        fTransVelocity = 0.;
        return fTransVelocity;
    }
    const double& KSTrajMagneticParticle::GetCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( fCharge ) * GetMagneticField().Magnitude()) / (2. * KConst::Pi() * GetLorentzFactor() * GetMass());
        return fCyclotronFrequency;
    }
    const double& KSTrajMagneticParticle::GetOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = 0.;
        return fOrbitalMagneticMoment;
    }

    //*****
    //cache
    //*****

    void KSTrajMagneticParticle::DoNothing() const
    {
        return;
    }
    void KSTrajMagneticParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetPosition(), GetTime(), fMagneticField );
        fGetMagneticFieldPtr = &KSTrajMagneticParticle::DoNothing;
        return;
    }
    void KSTrajMagneticParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetPosition(), GetTime(), fElectricField );
        fGetElectricFieldPtr = &KSTrajMagneticParticle::DoNothing;
        return;
    }
    void KSTrajMagneticParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fMagneticGradient );
        fGetMagneticGradientPtr = &KSTrajMagneticParticle::DoNothing;
        return;
    }
    void KSTrajMagneticParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotential );
        fGetElectricPotentialPtr = &KSTrajMagneticParticle::DoNothing;
        return;
    }

}

