#include "KSTrajAdiabaticParticle.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"
using katrin::KConst;

#include <cmath>

namespace Kassiopeia
{

    //0 is time
    //1 is length
    //2 is x component of guiding center
    //3 is y component of guiding center
    //4 is z component of guiding center
    //5 is longitudinal momentum
    //6 is transverse momentum
    //7 is phase

    KSMagneticField* KSTrajAdiabaticParticle::fMagneticFieldCalculator = NULL;
    KSElectricField* KSTrajAdiabaticParticle::fElectricFieldCalculator = NULL;
    double KSTrajAdiabaticParticle::fMass = 0.;
    double KSTrajAdiabaticParticle::fCharge = 0.;

    KSTrajAdiabaticParticle::KSTrajAdiabaticParticle() :
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
            fElectricPotentialRP( 0. ),

            fGuidingCenter( 0., 0., 0. ),
            fLongMomentum( 0. ),
            fTransMomentum( 0. ),
            fLongVelocity( 0. ),
            fTransVelocity( 0. ),
            fCyclotronFrequency( 0. ),
            fOrbitalMagneticMoment( 0. ),

            fAlpha( 0., 0., 0. ),
            fBeta( 0., 0., 0. ),
            fLastTime( 0.),
            fLastPosition( 0., 0., 0. ),
            fLastMomentum( 0., 0., 0. ),
            fPhase( 0. ),

            fGetMagneticFieldPtr( &KSTrajAdiabaticParticle::RecalculateMagneticField ),
            fGetElectricFieldPtr( &KSTrajAdiabaticParticle::RecalculateElectricField ),
            fGetMagneticGradientPtr( &KSTrajAdiabaticParticle::RecalculateMagneticGradient ),
            fGetMagneticFieldAndGradientPtr( &KSTrajAdiabaticParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialPtr( &KSTrajAdiabaticParticle::RecalculateElectricPotential ),
            fGetElectricPotentialRPPtr( &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP ),
            fGetElectricFieldAndPotentialPtr( &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential )

    {
    }
    KSTrajAdiabaticParticle::~KSTrajAdiabaticParticle()
    {
    }

    //**********
    //assignment
    //**********

    void KSTrajAdiabaticParticle::PullFrom( const KSParticle& aParticle )
    {
        trajmsg_debug( "adiabatic particle is pulling data from a particle..."<<eom);

        if( fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator() )
        {
            trajmsg_debug( "...magnetic calculator differs" )
            fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

            fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::RecalculateMagneticField;
            fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticGradient;
            fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient;
        }

        if( fElectricFieldCalculator != aParticle.GetElectricFieldCalculator() )
        {
            trajmsg_debug( "...electric calculator differs" )
            fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

            fGetElectricFieldPtr = &KSTrajAdiabaticParticle::RecalculateElectricField;
            fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotential;
            fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP;
            fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential;
        }

        if( GetMass() != aParticle.GetMass() )
        {
            trajmsg_debug( "...mass differs" )
            fMass = aParticle.GetMass();
        }

        if( GetCharge() != aParticle.GetCharge() )
        {
            trajmsg_debug( "...charge differs" )
            fCharge = aParticle.GetCharge();
        }

        if( fLastTime != aParticle.GetTime() || fLastPosition != aParticle.GetPosition() || fLastMomentum != aParticle.GetMomentum() )
        {
            fTime = aParticle.GetTime();
            fLength = aParticle.GetLength();
            fPosition = aParticle.GetPosition();
            fMomentum = aParticle.GetMomentum();
            fGuidingCenter = aParticle.GetGuidingCenterPosition();
            fMagneticFieldCalculator->CalculateField( fGuidingCenter, fTime, fMagneticField );

            fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
            fGetElectricFieldPtr = &KSTrajAdiabaticParticle::RecalculateElectricField;
            fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticGradient;
            fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient;
            fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotential;
            fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP;
            fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential;

            KThreeVector tGyrationVector = fGuidingCenter - fPosition;
            fAlpha = -1. * tGyrationVector.Unit();
            fBeta = -1. * fMagneticField.Cross( tGyrationVector ).Unit();

            fData[ 0 ] = fTime;
            fData[ 1 ] = fLength;
            fData[ 2 ] = fGuidingCenter.X();
            fData[ 3 ] = fGuidingCenter.Y();
            fData[ 4 ] = fGuidingCenter.Z();

            //renormalize momentum magnitude, this is necessary because the electric potential can
            //be different between the guiding center position and the particles true postion
            //since we have been given the particle's momentum, we have to correct the guiding center
            //momentum to account for this difference
            double tMC2 = fMass*KConst::C()*KConst::C();
            double tKineticEnergy = std::sqrt( fMomentum.MagnitudeSquared()*KConst::C()*KConst::C() + tMC2*tMC2 );
            //calculate potential at particles position
            double tRPPotential;
            fElectricFieldCalculator->CalculatePotential( fPosition, fTime, tRPPotential );
            //calulate potential at guiding center position
            double tGCPotential;
            fElectricFieldCalculator->CalculatePotential( fGuidingCenter, fTime, tGCPotential );
            tKineticEnergy -= (tGCPotential - tRPPotential)*fCharge;

            //need to take absolute value of sqrt argument to prevent nan's
            //if we are in a situation where the argument goes negative this probably means that
            //the guiding center approximation is not valid in that region
            double tMomentumMagnitude = (1.0/KConst::C())*std::sqrt( std::fabs( (tKineticEnergy - tMC2)*(tKineticEnergy + tMC2) ) );

            //TODO: Make sure we are not missing a correction term on the momentum due to difference in the magnetic
            //vector potential between the g.c and particle position
            fMomentum.SetMagnitude(tMomentumMagnitude);

            fData[ 5 ] = fMomentum.Dot( fMagneticField.Unit() );
            // though mathematically the expression inside the square root is guaranteed to be equal or bigger than 0
            // numerical errors can play us a trick and push it to negative values.
            // fabs() is therefore necessary to prevent everything to go bogus in that case.
            fData[ 6 ] = sqrt( fabs( (tMomentumMagnitude - fData[5])*(tMomentumMagnitude + fData[5])  ) );
            fData[ 7 ] = 0.;

			trajmsg_debug( "**updating adiabatic particle:" << ret )
			trajmsg_debug( "  real position: " << GetPosition() << ret )
			trajmsg_debug( "  real momentum: " << GetMomentum() << ret )
			trajmsg_debug( "  gc position: " << GetGuidingCenter() << ret )
			trajmsg_debug( "  gc alpha: " << fAlpha << ret )
			trajmsg_debug( "  gc beta: " << fBeta << ret )
			trajmsg_debug( "  parallel momentum: <" << GetLongMomentum() << ">" << ret )
			trajmsg_debug( "  perpendicular momentum: <" << GetTransMomentum() << ">" << ret )
			trajmsg_debug( "  kinetic energy is: <" << GetKineticEnergy()/ KConst::Q() << ">" << eom )

        }

        return;
    }
    void KSTrajAdiabaticParticle::PushTo( KSParticle& aParticle )
    {
    	trajmsg_debug( "AdiabaticParticle is pushing to KSParticle"<<ret);
		trajmsg_debug( "  real position: " << GetPosition() << ret )
		trajmsg_debug( "  real momentum: " << GetMomentum() << ret )
		trajmsg_debug( "  gc position: " << GetGuidingCenter() << ret )
		trajmsg_debug( "  gc alpha: " << fAlpha << ret )
		trajmsg_debug( "  gc beta: " << fBeta << ret )
		trajmsg_debug( "  parallel momentum: <" << GetLongMomentum() << ">" << ret )
		trajmsg_debug( "  perpendicular momentum: <" << GetTransMomentum() << ">" << ret )
		trajmsg_debug( "  kinetic energy is: <" << GetKineticEnergy()/ KConst::Q() << ">" << eom )

        fLastTime = GetTime();
        fLastPosition = GetPosition();
        fLastMomentum = GetMomentum();
        aParticle.SetTime( fLastTime );
        aParticle.SetPosition( fLastPosition );
        aParticle.SetMomentum( fLastMomentum );
        aParticle.SetLength( GetLength() );
        if( fGetElectricPotentialRPPtr == &KSTrajAdiabaticParticle::DoNothing )
        {
            aParticle.SetElectricPotential( GetElectricPotentialRP() );
        }
        fData[ 7 ] = fmod( fData[ 7 ], 2. * KConst::Pi() ); // you have to keep doing this to keep numerical precision in the angular variable

        return;
    }

    //***********
    //calculators
    //***********

    void KSTrajAdiabaticParticle::SetMagneticFieldCalculator( KSMagneticField* anMagneticField )
    {
        fMagneticFieldCalculator = anMagneticField;
        return;
    }
    KSMagneticField* KSTrajAdiabaticParticle::GetMagneticFieldCalculator()
    {
        return fMagneticFieldCalculator;
    }

    void KSTrajAdiabaticParticle::SetElectricFieldCalculator( KSElectricField* anElectricField )
    {
        fElectricFieldCalculator = anElectricField;
        return;
    }
    KSElectricField* KSTrajAdiabaticParticle::GetElectricFieldCalculator()
    {
        return fElectricFieldCalculator;
    }

    //****************
    //static variables
    //****************

    void KSTrajAdiabaticParticle::SetMass( const double& aMass )
    {
        fMass = aMass;
        return;
    }
    const double& KSTrajAdiabaticParticle::GetMass()
    {
        return fMass;
    }

    void KSTrajAdiabaticParticle::SetCharge( const double& aCharge )
    {
        fCharge = aCharge;
        return;
    }
    const double& KSTrajAdiabaticParticle::GetCharge()
    {
        return fCharge;
    }

    //*****************
    //dynamic variables
    //*****************

    const double& KSTrajAdiabaticParticle::GetTime() const
    {
        fTime = fData[ 0 ];
        return fTime;
    }
    const double& KSTrajAdiabaticParticle::GetLength() const
    {
		fLength = fData[ 1 ];
        return fLength;
    }
    const KThreeVector& KSTrajAdiabaticParticle::GetPosition() const
    {
        double tSigma = GetCharge() / fabs( GetCharge() );
        fPosition = GetGuidingCenter() + (fData[ 6 ] / (fabs( GetCharge() ) * GetMagneticField().Magnitude())) * (cos( tSigma * fData[ 7 ] ) * fAlpha + sin( tSigma * fData[ 7 ] ) * fBeta);
        return fPosition;
    }
    const KThreeVector& KSTrajAdiabaticParticle::GetMomentum() const
    {
        double tPhi = GetElectricPotentialRP() - GetElectricPotential();

        double tSigma = GetCharge() / fabs( GetCharge() );
        fMomentum = fData[ 5 ] * GetMagneticField().Unit() - fData[ 6 ] * tSigma * (-sin( tSigma * fData[ 7 ] ) * fAlpha + cos( tSigma * fData[ 7 ] ) * fBeta);

        double tPM = GetMass() * KConst::C();
        double tPPhi = (GetCharge() * tPhi) / KConst::C();
        double tPGC = sqrt( fData[ 5 ] * fData[ 5 ] + fData[ 6 ] * fData[ 6 ] );

        //need to take absolute value of sqrt argument to prevent nan's
        //if we are in a situation where the argument goes negative this probably means that
        //the guiding center approximation is not valid in that region
        double tPR = sqrt( std::fabs( tPGC * tPGC + tPPhi * tPPhi - 2. * tPPhi * sqrt( tPGC * tPGC + tPM * tPM ) ) );

        //TODO: Make sure we are not missing a correction term on the momentum due to difference in the magnetic
        //vector potential between the g.c and particle position

        fMomentum.SetMagnitude( tPR );

        return fMomentum;
    }
    const KThreeVector& KSTrajAdiabaticParticle::GetVelocity() const
    {
        fVelocity = (1. / (GetMass() * GetLorentzFactor())) * GetMomentum();
        return fVelocity;
    }
    const double& KSTrajAdiabaticParticle::GetLorentzFactor() const
    {
        fLorentzFactor = sqrt( 1. + (GetMomentum().MagnitudeSquared()) / (GetMass() * GetMass() * KConst::C() * KConst::C()) );
        return fLorentzFactor;
    }
    const double& KSTrajAdiabaticParticle::GetKineticEnergy() const
    {
        fKineticEnergy = (GetMomentum().MagnitudeSquared()) / ((1. + GetLorentzFactor()) * GetMass());
        return fKineticEnergy;
    }

    const KThreeVector& KSTrajAdiabaticParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldPtr)();
        return fMagneticField;
    }
    void KSTrajAdiabaticParticle::SetMagneticField( const KThreeVector& aField ) const
	{
    	fMagneticField = aField;
    	fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
	}
    const KThreeVector& KSTrajAdiabaticParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldPtr)();
        return fElectricField;
    }
    const KThreeMatrix& KSTrajAdiabaticParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientPtr)();
        return fMagneticGradient;
    }
    const std::pair<const KThreeVector&, const KThreeMatrix&> KSTrajAdiabaticParticle::GetMagneticFieldAndGradient() const
    {
        (this->*fGetMagneticFieldAndGradientPtr)();
        return std::make_pair(fMagneticField, fMagneticGradient );
    }
    const double& KSTrajAdiabaticParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialPtr)();
        return fElectricPotential;
    }
    const double& KSTrajAdiabaticParticle::GetElectricPotentialRP() const
    {
        (this->*fGetElectricPotentialRPPtr)();
        return fElectricPotentialRP;
    }
    const std::pair<const KThreeVector&, const double&> KSTrajAdiabaticParticle::GetElectricFieldAndPotential() const
    {
        (this->*fGetElectricFieldAndPotentialPtr)();
        return std::make_pair(fElectricField, fElectricPotential );
    }

    const KThreeVector& KSTrajAdiabaticParticle::GetGuidingCenter() const
    {
        fGuidingCenter.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fGuidingCenter;
    }
    const double& KSTrajAdiabaticParticle::GetLongMomentum() const
    {
        fLongMomentum = fData[ 5 ];
        return fLongMomentum;
    }
    const double& KSTrajAdiabaticParticle::GetTransMomentum() const
    {
        fTransMomentum = fData[ 6 ];
        return fTransMomentum;
    }
    const double& KSTrajAdiabaticParticle::GetLongVelocity() const
    {
        fLongVelocity = fData[ 5 ] / (GetMass() * GetLorentzFactor());
        return fLongVelocity;
    }
    const double& KSTrajAdiabaticParticle::GetTransVelocity() const
    {
        fTransVelocity = fData[ 6 ] / (GetMass() * GetLorentzFactor());
        return fTransVelocity;
    }
    const double& KSTrajAdiabaticParticle::GetCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( fCharge ) * GetMagneticField().Magnitude()) / (2. * KConst::Pi() * GetLorentzFactor() * GetMass());
        return fCyclotronFrequency;
    }
    const double& KSTrajAdiabaticParticle::GetOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = (fData[ 6 ] * fData[ 6 ]) / (2. * GetMagneticField().Magnitude() * GetMass());
        return fOrbitalMagneticMoment;
    }

    void KSTrajAdiabaticParticle::SetAlpha( const KThreeVector& anAlpha )
    {
        fAlpha = anAlpha;
        return;
    }
    const KThreeVector& KSTrajAdiabaticParticle::GetAlpha() const
    {
        return fAlpha;
    }

    void KSTrajAdiabaticParticle::SetBeta( const KThreeVector& aBeta )
    {
        fBeta = aBeta;
        return;
    }
    const KThreeVector& KSTrajAdiabaticParticle::GetBeta() const
    {
        return fBeta;
    }

    void KSTrajAdiabaticParticle::SetPhase( const double& aPhase )
    {
        fPhase = aPhase;
        return;
    }
    const double& KSTrajAdiabaticParticle::GetPhase() const
    {
    	fPhase = fData[ 7 ];
        return fPhase;
    }


    //*****
    //cache
    //*****

    void KSTrajAdiabaticParticle::DoNothing() const
    {
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetGuidingCenter(), GetTime(), fMagneticField );
        fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetGuidingCenter(), GetTime(), fElectricField );
        fGetElectricFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetGuidingCenter(), GetTime(), fMagneticGradient );
        fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient() const
    {
        //first check if either the magfield or the gradient are already cached
        //if one is cached, execute the function pointer of the other
        if ( fGetMagneticFieldPtr == &KSTrajAdiabaticParticle::DoNothing )
        {
            (this->*fGetMagneticGradientPtr)();
            fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::DoNothing;
            return;
        }
        if ( fGetMagneticGradientPtr == &KSTrajAdiabaticParticle::DoNothing )
        {
            (this->*fGetMagneticFieldPtr)();
            fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::DoNothing;
            return;
        }

        //if none is cached, calculate both at once
        fMagneticFieldCalculator->CalculateFieldAndGradient( GetGuidingCenter(), GetTime(), fMagneticField, fMagneticGradient );
        fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::DoNothing;
        fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
        fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetGuidingCenter(), GetTime(), fElectricPotential );
        fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateElectricPotentialRP() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotentialRP );
        fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }
    void KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential() const
    {
        //first check if either the electric field or the potential are already cached
        //if one is cached, execute the function pointer of the other
        if ( fGetElectricFieldPtr == &KSTrajAdiabaticParticle::DoNothing )
        {
            (this->*fGetElectricPotentialPtr)();
            fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::DoNothing;
            return;
        }
        if ( fGetElectricPotentialPtr == &KSTrajAdiabaticParticle::DoNothing )
        {
            (this->*fGetElectricFieldPtr)();
            fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::DoNothing;
            return;
        }

        //if none is cached, calculate both at once
        fElectricFieldCalculator->CalculateFieldAndPotential( GetGuidingCenter(), GetTime(), fElectricField, fElectricPotential );
        fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::DoNothing;
        fGetElectricFieldPtr = &KSTrajAdiabaticParticle::DoNothing;
        fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::DoNothing;
        return;
    }

} /* namespace Kassiopeia */
