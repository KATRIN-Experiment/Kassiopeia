#include "KSParticle.h"
#include "KSOperatorsMessage.h"

#include "KConst.h"
using katrin::KConst;

#include "KSDictionary.h"

#include <cmath>

namespace Kassiopeia
{

    const string KSParticle::sSeparator = string( ":" );

    //**********
    //assignment
    //**********

    KSParticle::KSParticle() :
            fLabel( "" ),
            fParentRunId( -1 ),
            fParentEventId( -1 ),
            fParentTrackId( -1 ),
            fParentStepId( -1 ),

            fActive( true ),
            fCurrentSpace( NULL ),
            fCurrentSurface( NULL ),
            fCurrentSide( NULL ),
            fCurrentSpaceName( "" ),
            fCurrentSurfaceName( "" ),
            fCurrentSideName( "" ),

            fMagneticFieldCalculator( NULL ),
            fElectricFieldCalculator( NULL ),

            fPID( 0 ),
            fMass( 0. ),
            fCharge( 0. ),
            fMoment( 0. ),

            fTime( 0. ),
            fLength( 0. ),
            fPosition( 0., 0., 0. ),
            fMomentum( 0., 0., 1. ),
            fVelocity( 0., 0., 0. ),
            fLorentzFactor( 0. ),
            fSpeed( 0. ),
            fKineticEnergy( 0. ),
            fKineticEnergy_eV( 0. ),
            fPolarAngleToZ( 0. ),
            fAzimuthalAngleToX( 0. ),

            fGetPositionAction( &KSParticle::DoNothing ),
            fGetMomentumAction( &KSParticle::DoNothing ),
            fGetVelocityAction( &KSParticle::RecalculateVelocity ),
            fGetLorentzFactorAction( &KSParticle::RecalculateLorentzFactor ),
            fGetSpeedAction( &KSParticle::RecalculateSpeed ),
            fGetKineticEnergyAction( &KSParticle::RecalculateKineticEnergy ),
            fGetPolarAngleToZAction( &KSParticle::RecalculatePolarAngleToZ ),
            fGetAzimuthalAngleToXAction( &KSParticle::RecalculateAzimuthalAngleToX ),

            fMagneticField( 0., 0., 0. ),
            fElectricField( 0., 0., 0. ),
            fMagneticGradient( 0., 0., 0., 0., 0., 0., 0., 0., 0. ),
            fElectricPotential( 0. ),

            fGetMagneticFieldAction( &KSParticle::RecalculateMagneticField ),
            fGetElectricFieldAction( &KSParticle::RecalculateElectricField ),
            fGetMagneticGradientAction( &KSParticle::RecalculateMagneticGradient ),
            fGetElectricPotentialAction( &KSParticle::RecalculateElectricPotential ),

            fLongMomentum( 0. ),
            fTransMomentum( 0. ),
            fLongVelocity( 0. ),
            fTransVelocity( 0. ),
            fPolarAngleToB( 0. ),
            fCyclotronFrequency( 0. ),
            fOrbitalMagneticMoment( 0. ),
            fGuidingCenterPosition( 0., 0., 0. ),

            fGetLongMomentumAction( &KSParticle::RecalculateLongMomentum ),
            fGetTransMomentumAction( &KSParticle::RecalculateTransMomentum ),
            fGetLongVelocityAction( &KSParticle::RecalculateLongVelocity ),
            fGetTransVelocityAction( &KSParticle::RecalculateTransVelocity ),
            fGetPolarAngleToBAction( &KSParticle::RecalculatePolarAngleToB ),
            fGetCyclotronFrequencyAction( &KSParticle::RecalculateCyclotronFrequency ),
            fGetOrbitalMagneticMomentAction( &KSParticle::RecalculateOrbitalMagneticMoment ),
            fGetGuidingCenterPositionAction( &KSParticle::RecalculateGuidingCenterPosition )
    {
    }
    KSParticle::KSParticle( const KSParticle& aParticle ) :
            fLabel( aParticle.fLabel ),
            fParentRunId( aParticle.fParentRunId ),
            fParentEventId( aParticle.fParentEventId ),
            fParentTrackId( aParticle.fParentTrackId ),
            fParentStepId( aParticle.fParentStepId ),

            fActive( aParticle.fActive ),
            fCurrentSpace( aParticle.fCurrentSpace ),
            fCurrentSurface( aParticle.fCurrentSurface ),
            fCurrentSide( aParticle.fCurrentSide ),
            fCurrentSpaceName( aParticle.fCurrentSpaceName ),
            fCurrentSurfaceName( aParticle.fCurrentSurfaceName ),
            fCurrentSideName( aParticle.fCurrentSideName ),

            fMagneticFieldCalculator( aParticle.fMagneticFieldCalculator ),
            fElectricFieldCalculator( aParticle.fElectricFieldCalculator ),

            fPID( aParticle.fPID ),
            fMass( aParticle.fMass ),
            fCharge( aParticle.fCharge ),
            fMoment( aParticle.fMoment ),

            fTime( aParticle.fTime ),
            fLength( aParticle.fLength ),
            fPosition( aParticle.fPosition ),
            fMomentum( aParticle.fMomentum ),
            fVelocity( aParticle.fVelocity ),
            fLorentzFactor( aParticle.fLorentzFactor ),
            fSpeed( aParticle.fSpeed ),
            fKineticEnergy( aParticle.fKineticEnergy ),
            fKineticEnergy_eV( aParticle.fKineticEnergy_eV ),
            fPolarAngleToZ( aParticle.fPolarAngleToZ ),
            fAzimuthalAngleToX( aParticle.fAzimuthalAngleToX ),

            fGetPositionAction( aParticle.fGetPositionAction ),
            fGetMomentumAction( aParticle.fGetMomentumAction ),
            fGetVelocityAction( aParticle.fGetVelocityAction ),
            fGetLorentzFactorAction( aParticle.fGetLorentzFactorAction ),
            fGetSpeedAction( aParticle.fGetSpeedAction ),
            fGetKineticEnergyAction( aParticle.fGetKineticEnergyAction ),
            fGetPolarAngleToZAction( aParticle.fGetPolarAngleToZAction ),
            fGetAzimuthalAngleToXAction( aParticle.fGetAzimuthalAngleToXAction ),

            fMagneticField( aParticle.fMagneticField ),
            fElectricField( aParticle.fElectricField ),
            fMagneticGradient( aParticle.fMagneticGradient ),
            fElectricPotential( aParticle.fElectricPotential ),

            fGetMagneticFieldAction( aParticle.fGetMagneticFieldAction ),
            fGetElectricFieldAction( aParticle.fGetElectricFieldAction ),
            fGetMagneticGradientAction( aParticle.fGetMagneticGradientAction ),
            fGetElectricPotentialAction( aParticle.fGetElectricPotentialAction ),

            fLongMomentum( aParticle.fLongMomentum ),
            fTransMomentum( aParticle.fTransMomentum ),
            fLongVelocity( aParticle.fLongVelocity ),
            fTransVelocity( aParticle.fTransVelocity ),
            fPolarAngleToB( aParticle.fPolarAngleToB ),
            fCyclotronFrequency( aParticle.fCyclotronFrequency ),
            fOrbitalMagneticMoment( aParticle.fOrbitalMagneticMoment ),
            fGuidingCenterPosition( aParticle.fGuidingCenterPosition ),

            fGetLongMomentumAction( aParticle.fGetLongMomentumAction ),
            fGetTransMomentumAction( aParticle.fGetTransMomentumAction ),
            fGetLongVelocityAction( aParticle.fGetLongVelocityAction ),
            fGetTransVelocityAction( aParticle.fGetTransVelocityAction ),
            fGetPolarAngleToBAction( aParticle.fGetPolarAngleToBAction ),
            fGetCyclotronFrequencyAction( aParticle.fGetCyclotronFrequencyAction ),
            fGetOrbitalMagneticMomentAction( aParticle.fGetOrbitalMagneticMomentAction ),
            fGetGuidingCenterPositionAction( aParticle.fGetGuidingCenterPositionAction )
    {
    }
    void KSParticle::operator=( const KSParticle& aParticle )
    {
        fLabel = aParticle.fLabel;
        fParentRunId = aParticle.fParentRunId;
        fParentEventId = aParticle.fParentEventId;
        fParentTrackId = aParticle.fParentTrackId;
        fParentStepId = aParticle.fParentStepId;

        fActive = aParticle.fActive;
        fCurrentSpace = aParticle.fCurrentSpace;
        fCurrentSurface = aParticle.fCurrentSurface;
        fCurrentSide = aParticle.fCurrentSide;
        fCurrentSpaceName = aParticle.fCurrentSpaceName;
        fCurrentSurfaceName = aParticle.fCurrentSurfaceName;
        fCurrentSideName = aParticle.fCurrentSideName;

        fMagneticFieldCalculator = aParticle.fMagneticFieldCalculator;
        fElectricFieldCalculator = aParticle.fElectricFieldCalculator;

        fPID = aParticle.fPID;
        fMass = aParticle.fMass;
        fCharge = aParticle.fCharge;
        fMoment = aParticle.fMoment;

        fTime = aParticle.fTime;
        fLength = aParticle.fLength;
        fPosition = aParticle.fPosition;
        fMomentum = aParticle.fMomentum;
        fVelocity = aParticle.fVelocity;
        fLorentzFactor = aParticle.fLorentzFactor;
        fSpeed = aParticle.fSpeed;
        fKineticEnergy = aParticle.fKineticEnergy;
        fPolarAngleToZ = aParticle.fPolarAngleToZ;
        fAzimuthalAngleToX = aParticle.fAzimuthalAngleToX;

        fGetPositionAction = aParticle.fGetPositionAction;
        fGetMomentumAction = aParticle.fGetMomentumAction;
        fGetVelocityAction = aParticle.fGetVelocityAction;
        fGetLorentzFactorAction = aParticle.fGetLorentzFactorAction;
        fGetSpeedAction = aParticle.fGetSpeedAction;
        fGetKineticEnergyAction = aParticle.fGetKineticEnergyAction;
        fGetPolarAngleToZAction = aParticle.fGetPolarAngleToZAction;
        fGetAzimuthalAngleToXAction = aParticle.fGetAzimuthalAngleToXAction;

        fMagneticField = aParticle.fMagneticField;
        fElectricField = aParticle.fElectricField;
        fMagneticGradient = aParticle.fMagneticGradient;
        fElectricPotential = aParticle.fElectricPotential;

        fGetMagneticFieldAction = aParticle.fGetMagneticFieldAction;
        fGetElectricFieldAction = aParticle.fGetElectricFieldAction;
        fGetMagneticGradientAction = aParticle.fGetMagneticGradientAction;
        fGetElectricPotentialAction = aParticle.fGetElectricPotentialAction;

        fLongMomentum = aParticle.fLongMomentum;
        fTransMomentum = aParticle.fTransMomentum;
        fLongVelocity = aParticle.fLongVelocity;
        fTransVelocity = aParticle.fTransVelocity;
        fPolarAngleToB = aParticle.fPolarAngleToB;
        fCyclotronFrequency = aParticle.fCyclotronFrequency;
        fOrbitalMagneticMoment = aParticle.fOrbitalMagneticMoment;
        fGuidingCenterPosition = aParticle.fGuidingCenterPosition;

        fGetLongMomentumAction = aParticle.fGetLongMomentumAction;
        fGetTransMomentumAction = aParticle.fGetTransMomentumAction;
        fGetLongVelocityAction = aParticle.fGetLongVelocityAction;
        fGetTransVelocityAction = aParticle.fGetTransVelocityAction;
        fGetPolarAngleToBAction = aParticle.fGetPolarAngleToBAction;
        fGetCyclotronFrequencyAction = aParticle.fGetCyclotronFrequencyAction;
        fGetOrbitalMagneticMomentAction = aParticle.fGetOrbitalMagneticMomentAction;
        fGetGuidingCenterPositionAction = aParticle.fGetGuidingCenterPositionAction;
    }
    KSParticle::~KSParticle()
    {
    }

    void KSParticle::DoNothing() const
    {
        return;
    }
    void KSParticle::Print() const
    {
        oprmsg( eNormal );
        oprmsg << "particle state:" << ret;
        oprmsg << "  id:         " << fPID << ret;
        oprmsg << "  mass:       " << fMass << ret;
        oprmsg << "  charge:     " << fCharge << ret;
        oprmsg << "  total spin: " << fMoment << ret;
        oprmsg << ret;
        oprmsg << "  x:          " << fPosition[ 0 ] << ret;
        oprmsg << "  y:          " << fPosition[ 1 ] << ret;
        oprmsg << "  z:          " << fPosition[ 2 ] << ret;
        oprmsg << "  px:         " << fMomentum[ 0 ] << ret;
        oprmsg << "  py:         " << fMomentum[ 1 ] << ret;
        oprmsg << "  pz:         " << fMomentum[ 2 ] << eom;
        return;
    }

    //******
    //labels
    //******

    void KSParticle::SetLabel( const string& aLabel )
    {
        fLabel = aLabel;
        return;
    }
    void KSParticle::AddLabel( const string& aLabel )
    {
        if( fLabel.size() != 0 )
        {
            fLabel.append( sSeparator );
        }
        fLabel.append( aLabel );
        return;
    }
    void KSParticle::ReleaseLabel( string& aLabel )
    {
        aLabel = fLabel;
        fLabel.clear();
        return;
    }

    void KSParticle::SetParentRunId( const int& anId )
    {
        fParentRunId = anId;
        return;
    }
    const int& KSParticle::GetParentRunId() const
    {
        return fParentRunId;
    }

    void KSParticle::SetParentEventId( const int& anId )
    {
        fParentEventId = anId;
        return;
    }
    const int& KSParticle::GetParentEventId() const
    {
        return fParentEventId;
    }

    void KSParticle::SetParentTrackId( const int& anId )
    {
        fParentTrackId = anId;
        return;
    }
    const int& KSParticle::GetParentTrackId() const
    {
        return fParentTrackId;
    }

    void KSParticle::SetParentStepId( const int& anId )
    {
        fParentStepId = anId;
        return;
    }
    const int& KSParticle::GetParentStepId() const
    {
        return fParentStepId;
    }

//*****
//state
//*****

    void KSParticle::SetActive( const bool& aFlag )
    {
        fActive = aFlag;
        return;
    }
    const bool& KSParticle::IsActive() const
    {
        return fActive;
    }

    void KSParticle::SetCurrentSpace( KSSpace* aSpace )
    {
        fCurrentSpace = aSpace;
        if ( fCurrentSpace != 0 )
        {
        	fCurrentSpaceName = fCurrentSpace->GetName();
        }
        else
        {
        	fCurrentSpaceName = string( "" );
        }
        return;
    }
    KSSpace* KSParticle::GetCurrentSpace() const
    {
        return fCurrentSpace;
    }
    const string& KSParticle::GetCurrentSpaceName() const
    {
    	return fCurrentSpaceName;
    }

    void KSParticle::SetCurrentSurface( KSSurface* aSurface )
    {
        fCurrentSurface = aSurface;
        if ( fCurrentSurface != 0 )
        {
        	fCurrentSurfaceName = fCurrentSurface->GetName();
        }
        else
        {
        	fCurrentSurfaceName = string( "" );
        }
        return;
    }
    KSSurface* KSParticle::GetCurrentSurface() const
    {
        return fCurrentSurface;
    }
    const string& KSParticle::GetCurrentSurfaceName() const
    {
    	return fCurrentSurfaceName;
    }

    void KSParticle::SetCurrentSide( KSSide* aSide )
    {
        fCurrentSide = aSide;
        if ( fCurrentSide != 0 )
        {
        	fCurrentSideName = fCurrentSide->GetName();
        }
        else
        {
        	fCurrentSideName = string( "" );
        }
        return;
    }
    KSSide* KSParticle::GetCurrentSide() const
    {
        return fCurrentSide;
    }
    const string& KSParticle::GetCurrentSideName() const
    {
    	return fCurrentSideName;
    }

//***********
//calculators
//***********

    void KSParticle::SetMagneticFieldCalculator( KSMagneticField* aMagFieldCalculator )
    {
        fMagneticFieldCalculator = aMagFieldCalculator;
        return;
    }
    KSMagneticField* KSParticle::GetMagneticFieldCalculator() const
    {
        return fMagneticFieldCalculator;
    }

    void KSParticle::SetElectricFieldCalculator( KSElectricField* const anElFieldCalculator )
    {
        fElectricFieldCalculator = anElFieldCalculator;
        return;
    }
    KSElectricField* KSParticle::GetElectricFieldCalculator() const
    {
        return fElectricFieldCalculator;
    }

//*****************
//static properties
//*****************

    const long long& KSParticle::GetPID() const
    {
        return fPID;
    }
    const double& KSParticle::GetMass() const
    {
        return fMass;
    }
    const double& KSParticle::GetCharge() const
    {
        return fCharge;
    }
    const double& KSParticle::GetMoment() const
    {
        return fMoment;
    }

//****
//time
//****

    void KSParticle::SetTime( const double& t )
    {
        fTime = t;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fTime" << ret );
        //oprmsg_debug( "[" << fTime << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;

    }
    const double& KSParticle::GetTime() const
    {
        return fTime;
    }

//******
//length
//******

    void KSParticle::SetLength( const double& l )
    {
        fLength = l;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fLength" << ret );
        //oprmsg_debug( "[" << fLength << "]" << eom );

        return;
    }
    const double& KSParticle::GetLength() const
    {
        return fLength;
    }

//********
//position
//********

    const KThreeVector& KSParticle::GetPosition() const
    {
        (this->*fGetPositionAction)();
        return fPosition;
    }
    double KSParticle::GetX() const
    {
        (this->*fGetPositionAction)();
        return fPosition[ 0 ];
    }
    double KSParticle::GetY() const
    {
        (this->*fGetPositionAction)();
        return fPosition[ 1 ];
    }
    double KSParticle::GetZ() const
    {
        (this->*fGetPositionAction)();
        return fPosition[ 2 ];
    }

    void KSParticle::SetPosition( const KThreeVector& aPosition )
    {
        fPosition = aPosition;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPosition" << ret );
        //oprmsg_debug( "[" << fPosition[0] << ", " << fPosition[1] << ", " << fPosition[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetPosition( const double& anX, const double& aY, const double& aZ )
    {
        fPosition.SetComponents( anX, aY, aZ );

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPosition" << ret );
        //oprmsg_debug( "[" << fPosition[0] << ", " << fPosition[1] << ", " << fPosition[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetX( const double& anX )
    {
        fPosition[ 0 ] = anX;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPosition" << ret );
        //oprmsg_debug( "[" << fPosition[0] << ", " << fPosition[1] << ", " << fPosition[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetY( const double& aY )
    {
        fPosition[ 1 ] = aY;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPosition" << ret );
        //oprmsg_debug( "[" << fPosition[0] << ", " << fPosition[1] << ", " << fPosition[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;
        return;
    }
    void KSParticle::SetZ( const double& aZ )
    {
        fPosition[ 2 ] = aZ;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPosition" << ret );
        //oprmsg_debug( "[" << fPosition[0] << ", " << fPosition[1] << ", " << fPosition[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        fGetElectricFieldAction = &KSParticle::RecalculateElectricField;
        fGetMagneticGradientAction = &KSParticle::RecalculateMagneticGradient;
        fGetElectricPotentialAction = &KSParticle::RecalculateElectricPotential;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculatePosition() const
    {
        //this function should not be called since position is a basic variable
        return;
    }

//********
//momentum
//********

    const KThreeVector& KSParticle::GetMomentum() const
    {

        //oprmsg_debug( "KSParticle: [" << this << "] getting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        (this->*fGetMomentumAction)();
        return fMomentum;
    }
    double KSParticle::GetPX() const
    {
        (this->*fGetMomentumAction)();
        return fMomentum[ 0 ];
    }
    double KSParticle::GetPY() const
    {
        (this->*fGetMomentumAction)();
        return fMomentum[ 1 ];
    }
    double KSParticle::GetPZ() const
    {
        (this->*fGetMomentumAction)();
        return fMomentum[ 2 ];
    }

    void KSParticle::SetMomentum( const KThreeVector& aMomentum )
    {
        fMomentum = aMomentum;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetMomentum( const double& anX, const double& aY, const double& aZ )
    {
        fMomentum.SetComponents( anX, aY, aZ );

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetPX( const double& anX )
    {
        fMomentum[ 0 ] = anX;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetPY( const double& aY )
    {
        fMomentum[ 1 ] = aY;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::SetPZ( const double& aZ )
    {
        fMomentum[ 2 ] = aZ;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMomentum" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateMomentum() const
    {
        //this function should not be called since momentum is a basic variable
        return;
    }

//********
//velocity
//********

    const KThreeVector& KSParticle::GetVelocity() const
    {
        (this->*fGetVelocityAction)();
        return fVelocity;
    }

    void KSParticle::SetVelocity( const KThreeVector& NewVelocity )
    {
        double Speed = NewVelocity.Magnitude();
        double LorentzFactor = 1.0 / sqrt( 1.0 - (Speed * Speed / (KConst::C() * KConst::C())) );

        fMomentum = GetMass() * LorentzFactor * NewVelocity;
        fVelocity = NewVelocity;
        fLorentzFactor = LorentzFactor;
        fSpeed = Speed;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fVelocity" << ret );
        //oprmsg_debug( "[" << fVelocity[0] << ", " << fVelocity[1] << ", " << fVelocity[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fLorentzFactor and fSpeed have been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << ret );
        //oprmsg_debug( "[" << fSpeed << "]" << eom );

        fGetVelocityAction = &KSParticle::DoNothing;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::DoNothing;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateVelocity() const
    {
        fVelocity = (1. / (GetMass() * GetLorentzFactor())) * fMomentum;

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fVelocity" << ret );
        //oprmsg_debug( "[" << fVelocity[0] << ", " << fVelocity[1] << ", " << fVelocity[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::DoNothing;

        return;
    }

//*****
//speed
//*****

    const double& KSParticle::GetSpeed() const
    {
        (this->*fGetSpeedAction)();
        return fSpeed;
    }

    void KSParticle::SetSpeed( const double& NewSpeed )
    {
        double LorentzFactor = 1.0 / sqrt( 1.0 - NewSpeed * NewSpeed / (KConst::C() * KConst::C()) );
        double MomentumMagnitude = GetMass() * KConst::C() * sqrt( LorentzFactor * LorentzFactor - 1.0 );

        fMomentum.SetMagnitude( MomentumMagnitude );
        fLorentzFactor = LorentzFactor;
        fSpeed = NewSpeed;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fSpeed" << ret );
        //oprmsg_debug( "[" << fSpeed << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fLorentzFactor has been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::DoNothing;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateSpeed() const
    {
        fSpeed = GetVelocity().Magnitude();

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fSpeed" << ret );
        //oprmsg_debug( "[" << fSpeed << "]" << eom );

        fGetSpeedAction = &KSParticle::DoNothing;

        return;
    }

//**************
//lorentz factor
//**************

    const double& KSParticle::GetLorentzFactor() const
    {
        (this->*fGetLorentzFactorAction)();
        return fLorentzFactor;
    }

    void KSParticle::SetLorentzFactor( const double& NewLorentzFactor )
    {
        double MomentumMagnitude = GetMass() * KConst::C() * sqrt( NewLorentzFactor * NewLorentzFactor - 1.0 );

        fMomentum.SetMagnitude( MomentumMagnitude );
        fLorentzFactor = NewLorentzFactor;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fLorentzFactor" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateLorentzFactor() const
    {
        fLorentzFactor = sqrt( 1.0 + fMomentum.MagnitudeSquared() / (GetMass() * GetMass() * KConst::C() * KConst::C()) );

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fLorentzFactor" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << eom );

        fGetLorentzFactorAction = &KSParticle::DoNothing;

        return;
    }

//**************
//kinetic energy
//**************

    const double& KSParticle::GetKineticEnergy() const
    {
        (this->*fGetKineticEnergyAction)();
        return fKineticEnergy;
    }

    void KSParticle::SetKineticEnergy( const double& NewKineticEnergy )
    {
        double MomentumMagnitude = (NewKineticEnergy / KConst::C()) * sqrt( 1.0 + (2.0 * GetMass() * KConst::C() * KConst::C()) / NewKineticEnergy );
        fMomentum.SetMagnitude( MomentumMagnitude );
        fKineticEnergy = NewKineticEnergy;

        //utilmsg(eDebug) << "KSParticle: [" << this << "] setting fKineticEnergy" << ret );
        //oprmsg_debug( "[" << fKineticEnergy << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::DoNothing;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateKineticEnergy() const
    {
        fKineticEnergy = fMomentum.MagnitudeSquared() / ((1.0 + GetLorentzFactor()) * GetMass());

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fKineticEnergy" << ret );
        //oprmsg_debug( "momentum.mag2: [" << fMomentum.MagnitudeSquared() << "]" << eom );
        //oprmsg_debug( "lorentzfactor: [" << GetLorentzFactor() << "]" << eom );
        //oprmsg_debug( "mass: [" << GetMass() << "]" << eom );
        //oprmsg_debug( "kinetic energy: [" << fKineticEnergy << "]" << eom );

        fGetKineticEnergyAction = &KSParticle::DoNothing;

        return;
    }

//setter and getter for KineticEnergy in eV

    void KSParticle::SetKineticEnergy_eV( const double& NewKineticEnergy )
    {

        //oprmsg_debug( "KSParticle: [" << this << "] setting fKineticEnergy in eV" << ret );
        //oprmsg_debug( "[" << NewKineticEnergy << "]" << eom );

        double NewKineticEnergy_SI = NewKineticEnergy * KConst::Q();
        SetKineticEnergy( NewKineticEnergy_SI );
        return;
    }

    const double& KSParticle::GetKineticEnergy_eV() const
    {
        fKineticEnergy_eV = GetKineticEnergy() / KConst::Q();

        //oprmsg_debug( "KSParticle: [" << this << "] getting fKineticEnergy in eV" << ret );
        //oprmsg_debug( "[" << fKineticEnergy_eV << "]" << eom );

        return fKineticEnergy_eV;
    }

//****************
//polar angle to z
//****************

    const double& KSParticle::GetPolarAngleToZ() const
    {
        (this->*fGetPolarAngleToZAction)();
        return fPolarAngleToZ;
    }

    void KSParticle::SetPolarAngleToZ( const double& NewPolarAngleToZ )
    {
        if( (NewPolarAngleToZ < 0.0) || (NewPolarAngleToZ > 180.0) )
        {
            //oprmsg( eWarning ) << " Polar Angle is only defined between 0 and 180 " << eom;
        }
        fPolarAngleToZ = NewPolarAngleToZ;

        double MomentumMagnitude = fMomentum.Magnitude();
        double NewPolarAngleToZ_SI = fPolarAngleToZ / 180. * KConst::Pi();

        fMomentum.SetComponents( MomentumMagnitude * sin( NewPolarAngleToZ_SI ) * cos( GetAzimuthalAngleToX() ), MomentumMagnitude * sin( NewPolarAngleToZ_SI ) * sin( GetAzimuthalAngleToX() ), MomentumMagnitude * cos( NewPolarAngleToZ_SI ) );

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPolarAngleToZ" << ret );
        //oprmsg_debug( "[" << fPolarAngleToZ << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        //fGetLorentzFactorAction unchanged
        //fGetSpeedAction unchanged
        //fGetKineticEnergyAction unchanged
        fGetPolarAngleToZAction = &KSParticle::DoNothing;
        //fGetAzimuthalAngleToXAction unchanged

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        //fGetCyclotronFrequencyAction unchanged
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculatePolarAngleToZ() const
    {
        fPolarAngleToZ = (180. / KConst::Pi()) * acos( fMomentum[ 2 ] / fMomentum.Magnitude() );

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fPolarAngleToZ" << ret );
        //oprmsg_debug( "[" << fPolarAngleToZ << "]" << eom );

        fGetPolarAngleToZAction = &KSParticle::DoNothing;

        return;
    }

//********************
//azimuthal angle to x
//********************

    const double& KSParticle::GetAzimuthalAngleToX() const
    {
        (this->*fGetAzimuthalAngleToXAction)();
        return fAzimuthalAngleToX;
    }

    void KSParticle::SetAzimuthalAngleToX( const double& NewAzimuthalAngleToX )
    {
        fAzimuthalAngleToX = NewAzimuthalAngleToX;
        if( fAzimuthalAngleToX < 0.0 )
        {
            fAzimuthalAngleToX += 360.0;
        }

        double NewAzimuthalAngleToX_SI = (KConst::Pi() / 180.) * fAzimuthalAngleToX;
        double PolarAngleToZ_SI = (KConst::Pi() / 180.) * GetPolarAngleToZ();
        double MomentumMagnitude = fMomentum.Magnitude();

        fMomentum.SetComponents( MomentumMagnitude * sin( PolarAngleToZ_SI ) * cos( NewAzimuthalAngleToX_SI ), MomentumMagnitude * sin( PolarAngleToZ_SI ) * sin( NewAzimuthalAngleToX_SI ), MomentumMagnitude * cos( PolarAngleToZ_SI ) );

        //oprmsg_debug( "KSParticle: [" << this << "] setting fAzimuthalAngleToX" << ret );
        //oprmsg_debug( "[" << fAzimuthalAngleToX << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        //fGetLorentzFactorAction unchanged
        //fGetSpeedAction unchanged
        //fGetKineticEnergyAction unchanged
        //fGetPolarAngleToZAction unchanged
        fGetAzimuthalAngleToXAction = &KSParticle::DoNothing;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        //fGetCyclotronFrequencyAction unchanged
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }

    void KSParticle::RecalculateAzimuthalAngleToX() const
    {
        register double Px = GetMomentum()[ 0 ];
        register double Py = GetMomentum()[ 1 ];
        register double NewCosPhi = 0;

        if( Px == 0. )
        {
            if( Py == 0. )
            {
                fAzimuthalAngleToX = 0.;
            }
            else if( Py >= 0. )
            {
                fAzimuthalAngleToX = 90.;
            }
            else
            {
                fAzimuthalAngleToX = 270.;
            }
        }
        else
        {
            NewCosPhi = Px / sqrt( Px * Px + Py * Py );
            if( GetMomentum()[ 1 ] >= 0. )
            {
                fAzimuthalAngleToX = (180. / KConst::Pi()) * acos( NewCosPhi );
            }
            else
            {
                fAzimuthalAngleToX = (180. / KConst::Pi()) * (2. * KConst::Pi() - acos( NewCosPhi ));
            }
        }

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fAzimuthalAngleToX" << ret );
        //oprmsg_debug( "[" << fAzimuthalAngleToX << "]" << eom );

        fGetAzimuthalAngleToXAction = &KSParticle::DoNothing;

        return;
    }

//**************
//magnetic field
//**************

    const KThreeVector& KSParticle::GetMagneticField() const
    {
        (this->*fGetMagneticFieldAction)();
        return fMagneticField;
    }

    void KSParticle::SetMagneticField( const KThreeVector& aMagneticField )
    {
        fMagneticField = aMagneticField;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMagneticField" << ret );
        //oprmsg_debug( "[" << fMagneticField[0] << ", " << fMagneticField[1] << ", " << fMagneticField[2] << "]" << eom );

        fGetMagneticFieldAction = &KSParticle::DoNothing;

        return;
    }

    void KSParticle::RecalculateMagneticField() const
    {
        fMagneticFieldCalculator->CalculateField( GetPosition(), GetTime(), fMagneticField );
        fGetMagneticFieldAction = &KSParticle::DoNothing;
        return;
    }

//**************
//electric field
//**************

    const KThreeVector& KSParticle::GetElectricField() const
    {
        (this->*fGetElectricFieldAction)();
        return fElectricField;
    }

    void KSParticle::SetElectricField( const KThreeVector& aElectricField )
    {
        fElectricField = aElectricField;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fElectricField" << ret );
        //oprmsg_debug( "[" << fElectricField[0] << ", " << fElectricField[1] << ", " << fElectricField[2] << "]" << eom );

        fGetElectricFieldAction = &KSParticle::DoNothing;

        return;
    }

    void KSParticle::RecalculateElectricField() const
    {
        fElectricFieldCalculator->CalculateField( GetPosition(), GetTime(), fElectricField );
        fGetElectricFieldAction = &KSParticle::DoNothing;
        return;
    }

//***********************
//magnetic field gradient
//***********************

    const KThreeMatrix& KSParticle::GetMagneticGradient() const
    {
        (this->*fGetMagneticGradientAction)();
        return fMagneticGradient;
    }

    void KSParticle::SetMagneticGradient( const KThreeMatrix& aMagneticGradient )
    {
        fMagneticGradient = aMagneticGradient;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fMagneticGradient" << eom );
        ////oprmsg_debug( "[" << fMagneticGradient << "]" << eom );

        return;
    }

    void KSParticle::RecalculateMagneticGradient() const
    {
        fMagneticFieldCalculator->CalculateGradient( GetPosition(), GetTime(), fMagneticGradient );
        fGetMagneticGradientAction = &KSParticle::DoNothing;
        return;
    }

//******************
//electric potential
//******************

    const double& KSParticle::GetElectricPotential() const
    {
        (this->*fGetElectricPotentialAction)();
        return fElectricPotential;
    }

    void KSParticle::SetElectricPotential( const double& anElectricPotential )
    {
        fElectricPotential = anElectricPotential;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fElectricPotential" << ret );
        //oprmsg_debug( "[" << fElectricPotential << "]" << eom );

        fGetElectricPotentialAction = &KSParticle::DoNothing;

        return;
    }

    void KSParticle::RecalculateElectricPotential() const
    {
        fElectricFieldCalculator->CalculatePotential( GetPosition(), GetTime(), fElectricPotential );
        fGetElectricPotentialAction = &KSParticle::DoNothing;
        return;
    }

//*********************
//longitudinal momentum
//*********************

    void KSParticle::SetLongMomentum( const double& aNewLongMomentum )
    {
        KThreeVector LongMomentumVector = GetMagneticField();
        LongMomentumVector.SetMagnitude( aNewLongMomentum - GetLongMomentum() );

        fMomentum += LongMomentumVector;
        fLongMomentum = aNewLongMomentum;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fLongMomentum" << ret );
        //oprmsg_debug( "[" << fLongMomentum << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::DoNothing;
        //fGetTransMomentumAction unchanged
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        //fGetOrbitalMagneticMomentAction unchanged

        return;
    }
    const double& KSParticle::GetLongMomentum() const
    {
        (this->*fGetLongMomentumAction)();
        return fLongMomentum;
    }
    void KSParticle::RecalculateLongMomentum() const
    {
        fLongMomentum = GetMagneticField().Unit().Dot( fMomentum );

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fLongMomentum" << ret );
        //oprmsg_debug( "[" << fLongMomentum << "]" << eom );

        fGetLongMomentumAction = &KSParticle::DoNothing;

        return;
    }

//*******************
//transverse momentum
//*******************

    void KSParticle::SetTransMomentum( const double& NewTransMomentum )
    {
        KThreeVector LongMomentumVector = GetLongMomentum() * GetMagneticField().Unit();
        KThreeVector TransMomentumVector = fMomentum - LongMomentumVector;
        TransMomentumVector.SetMagnitude( NewTransMomentum );

        fMomentum = LongMomentumVector + TransMomentumVector;
        fTransMomentum = NewTransMomentum;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fTransMomentum" << ret );
        //oprmsg_debug( "[" << fTransMomentum << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        //fGetLongMomentumAction unchanged
        fGetTransMomentumAction = &KSParticle::DoNothing;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    const double& KSParticle::GetTransMomentum() const
    {
        (this->*fGetTransMomentumAction)();
        return fTransMomentum;
    }
    void KSParticle::RecalculateTransMomentum() const
    {
        fTransMomentum = sqrt( fMomentum.MagnitudeSquared() - GetLongMomentum() * GetLongMomentum() );

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fTransMomentum" << ret );
        //oprmsg_debug( "[" << fTransMomentum << "]" << eom );

        fGetTransMomentumAction = &KSParticle::DoNothing;

        return;
    }

//*********************
//longitudinal velocity
//*********************

    void KSParticle::SetLongVelocity( const double& NewLongVelocity )
    {
        KThreeVector LongVelocityVector = GetMagneticField();
        LongVelocityVector.SetMagnitude( NewLongVelocity - GetLongVelocity() );

        fVelocity += LongVelocityVector;
        fSpeed = fVelocity.Magnitude();
        fLorentzFactor = 1.0 / sqrt( 1.0 - (fSpeed * fSpeed / (KConst::C() * KConst::C())) );
        fMomentum = (GetMass() * fLorentzFactor) * fVelocity;
        fLongVelocity = NewLongVelocity;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fLongVelocity" << ret );
        //oprmsg_debug( "[" << fLongVelocity << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fVelocity, fLorentzFactor and fSpeed have been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fVelocity[0] << ", " << fVelocity[1] << ", " << fVelocity[2] << "]" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << ret );
        //oprmsg_debug( "[" << fSpeed << "]" << eom );

        fGetVelocityAction = &KSParticle::DoNothing;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::DoNothing;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::DoNothing;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    const double& KSParticle::GetLongVelocity() const
    {
        (this->*fGetLongVelocityAction)();
        return fLongVelocity;
    }
    void KSParticle::RecalculateLongVelocity() const
    {
        fLongVelocity = GetLongMomentum() / (GetMass() * GetLorentzFactor());

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fLongVelocity" << ret );
        //oprmsg_debug( "[" << fLongVelocity << "]" << eom );

        fGetLongVelocityAction = &KSParticle::DoNothing;

        return;
    }

//*******************
//transverse velocity
//*******************

    void KSParticle::SetTransVelocity( const double& NewTransVelocity )
    {
        KThreeVector TransVelocityVector = GetVelocity() - GetLongVelocity() * GetMagneticField().Unit();
        TransVelocityVector.SetMagnitude( NewTransVelocity - GetTransVelocity() );

        fVelocity += TransVelocityVector;
        fSpeed = fVelocity.Magnitude();
        fLorentzFactor = 1.0 / sqrt( 1.0 - (fSpeed * fSpeed / (KConst::C() * KConst::C())) );
        fMomentum = (GetMass() * fLorentzFactor) * fVelocity;
        fTransVelocity = NewTransVelocity;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fTransVelocity" << ret );
        //oprmsg_debug( "[" << fTransVelocity << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fVelocity, fLorentzFactor and fSpeed have been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fVelocity[0] << ", " << fVelocity[1] << ", " << fVelocity[2] << "]" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << ret );
        //oprmsg_debug( "[" << fSpeed << "]" << eom );

        fGetVelocityAction = &KSParticle::DoNothing;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::DoNothing;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::DoNothing;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    const double& KSParticle::GetTransVelocity() const
    {
        (this->*fGetTransVelocityAction)();
        return fTransVelocity;
    }
    void KSParticle::RecalculateTransVelocity() const
    {
        fTransVelocity = GetTransMomentum() / (GetMass() * GetLorentzFactor());

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fTransVelocity" << ret );
        //oprmsg_debug( "[" << fTransVelocity << "]" << eom );

        fGetTransVelocityAction = &KSParticle::DoNothing;

        return;
    }

//****************
//polar angle to B
//****************

    void KSParticle::SetPolarAngleToB( const double& NewPolarAngleToB )
    {
        if( (NewPolarAngleToB < 0.0) || (NewPolarAngleToB > 180.0) )
        {
            //oprmsg( eWarning ) << "Polar angle is only defined between 0 and 180 degree" << eom;
        }
        double NewPolarAngleToB_SI = (KConst::Pi() / 180.) * NewPolarAngleToB;
        double MomentumMagnitude = fMomentum.Magnitude();
        KThreeVector LongUnit = GetMagneticField().Unit();
        KThreeVector TransUnit = (fMomentum - GetLongMomentum() * LongUnit).Unit();

        fMomentum = MomentumMagnitude * cos( NewPolarAngleToB_SI ) * LongUnit + MomentumMagnitude * sin( NewPolarAngleToB_SI ) * TransUnit;
        fPolarAngleToB = NewPolarAngleToB;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fPolarAngleToB" << ret );
        //oprmsg_debug( "[" << fPolarAngleToB << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        //fGetLorentzFactorAction unchanged
        //fGetSpeedAction unchanged
        //fGetKineticEnergyAction unchanged
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::DoNothing;
        //fGetCyclotronFrequencyAction unchanged
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    const double& KSParticle::GetPolarAngleToB() const
    {
        (this->*fGetPolarAngleToBAction)();
        return fPolarAngleToB;
    }
    void KSParticle::RecalculatePolarAngleToB() const
    {
        fPolarAngleToB = acos( fMomentum.Unit().Dot( GetMagneticField().Unit() ) ) * 180. / KConst::Pi();

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fPolarAngleToB" << ret );
        //oprmsg_debug( "[" << fPolarAngleToB << "]" << eom );

        fGetPolarAngleToBAction = &KSParticle::DoNothing;

        return;
    }

//*******************
//cyclotron frequency
//*******************

    void KSParticle::SetCyclotronFrequency( const double& NewCyclotronFrequency )
    {
        double LorentzFactor = (GetCharge() * GetMagneticField().Magnitude()) / (2.0 * KConst::Pi() * GetMass() * NewCyclotronFrequency);
        double MomentumMagnitude = GetMass() * KConst::C() * sqrt( LorentzFactor * LorentzFactor - 1.0 );

        fMomentum.SetMagnitude( MomentumMagnitude );
        fLorentzFactor = LorentzFactor;
        fCyclotronFrequency = NewCyclotronFrequency;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fCyclotronFrequency" << ret );
        //oprmsg_debug( "[" << fCyclotronFrequency << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fLorentzFactor has been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fLorentzFactor << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::DoNothing;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        fGetLongMomentumAction = &KSParticle::RecalculateLongMomentum;
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::DoNothing;
        fGetOrbitalMagneticMomentAction = &KSParticle::RecalculateOrbitalMagneticMoment;

        return;
    }
    void KSParticle::RecalculateCyclotronFrequency() const
    {
        fCyclotronFrequency = (fabs( GetCharge() ) * GetMagneticField().Magnitude()) / (2.0 * KConst::Pi() * GetMass() * GetLorentzFactor());

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fCyclotronFrequency" << ret );
        //oprmsg_debug( "[" << fCyclotronFrequency << "]" << eom );

        fGetCyclotronFrequencyAction = &KSParticle::DoNothing;

        return;
    }
    const double& KSParticle::GetCyclotronFrequency() const
    {
        (this->*fGetCyclotronFrequencyAction)();
        return fCyclotronFrequency;
    }

//***********************
//orbital magnetic moment
//***********************

    const double& KSParticle::GetOrbitalMagneticMoment() const
    {

        (this->*fGetOrbitalMagneticMomentAction)();
        return fOrbitalMagneticMoment;
    }
    void KSParticle::SetOrbitalMagneticMoment( const double& NewOrbitalMagneticMoment )
    {
        double TransMomentumMagnitude = sqrt( 2.0 * GetMass() * GetMagneticField().Magnitude() * NewOrbitalMagneticMoment );
        KThreeVector TransMomentumVector = fMomentum - GetLongMomentum() * GetMagneticField().Unit();
        TransMomentumVector.SetMagnitude( TransMomentumMagnitude - GetTransMomentum() );

        fMomentum += TransMomentumVector;
        fTransMomentum = TransMomentumMagnitude;
        fOrbitalMagneticMoment = NewOrbitalMagneticMoment;

        //oprmsg_debug( "KSParticle: [" << this << "] setting fOrbitalMagneticMoment" << ret );
        //oprmsg_debug( "[" << fOrbitalMagneticMoment << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fMomentum has been recalculated" << ret );
        //oprmsg_debug( "[" << fMomentum[0] << ", " << fMomentum[1] << ", " << fMomentum[2] << "]" << ret );
        //oprmsg_debug( "KSParticle: [" << this << "] fTransMomentum has been secondarily recalculated" << ret );
        //oprmsg_debug( "[" << fTransMomentum << "]" << eom );

        fGetVelocityAction = &KSParticle::RecalculateVelocity;
        fGetLorentzFactorAction = &KSParticle::RecalculateLorentzFactor;
        fGetSpeedAction = &KSParticle::RecalculateSpeed;
        fGetKineticEnergyAction = &KSParticle::RecalculateKineticEnergy;
        fGetPolarAngleToZAction = &KSParticle::RecalculatePolarAngleToZ;
        fGetAzimuthalAngleToXAction = &KSParticle::RecalculateAzimuthalAngleToX;

        //fGetLongMomentumAction unchanged
        fGetTransMomentumAction = &KSParticle::RecalculateTransMomentum;
        fGetLongVelocityAction = &KSParticle::RecalculateLongVelocity;
        fGetTransVelocityAction = &KSParticle::RecalculateTransVelocity;
        fGetPolarAngleToBAction = &KSParticle::RecalculatePolarAngleToB;
        fGetCyclotronFrequencyAction = &KSParticle::RecalculateCyclotronFrequency;
        fGetOrbitalMagneticMomentAction = &KSParticle::DoNothing;

        return;
    }
    void KSParticle::RecalculateOrbitalMagneticMoment() const
    {
        fOrbitalMagneticMoment = (GetTransMomentum() * GetTransMomentum()) / (2.0 * GetMass() * GetMagneticField().Magnitude());

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fOrbitalMagneticMoment" << ret );
        //oprmsg_debug( "[" << fOrbitalMagneticMoment << "]" << eom );

        fGetOrbitalMagneticMomentAction = &KSParticle::DoNothing;

        return;
    }

//***********************
//guiding center position
//***********************

//TODO: incorporate this into the caching system, figure out correct setter for gc position!

    const KThreeVector& KSParticle::GetGuidingCenterPosition() const
    {
        (this->*fGetGuidingCenterPositionAction)();
        return fGuidingCenterPosition;
    }
    void KSParticle::SetGuidingCenterPosition( const KThreeVector& NewGuidingCenterPosition )
    {
        fGuidingCenterPosition = NewGuidingCenterPosition;
    }
    void KSParticle::RecalculateGuidingCenterPosition() const
    {
        fGuidingCenterPosition = GetPosition() + (1.0 / (GetCharge() * GetMagneticField().MagnitudeSquared())) * (GetMomentum().Cross( GetMagneticField() ));

        //calculate magnetic field at gc position
        KThreeVector tRealPosition = GetPosition();
        fPosition = fGuidingCenterPosition;
        KThreeVector tRealMagneticField = GetMagneticField();
        fGetMagneticFieldAction = &KSParticle::RecalculateMagneticField;
        KThreeVector tGCMagneticField = GetMagneticField();

        //calculate guiding center position again, now with better magnetic field
        fGuidingCenterPosition = tRealPosition + (1.0 / (GetCharge() * tGCMagneticField.MagnitudeSquared())) * (GetMomentum().Cross( tGCMagneticField ));

        //set back old cached values for position and magneticfield
        fPosition = tRealPosition;
        fMagneticField = tRealMagneticField;

        //oprmsg_debug( "KSParticle: [" << this << "] recalculating fGuidingCenterPosition" << ret );
        //oprmsg_debug( "[" << fGuidingCenterPosition[0] << ", " << fGuidingCenterPosition[1] << ", " << fGuidingCenterPosition[2] << "]" << eom );

        fGetGuidingCenterPositionAction = &KSParticle::DoNothing;

        return;
    }

    static const int sKSParticleDict =
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetParentRunId, "parent_run_id" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetParentEventId, "parent_event_id" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetParentTrackId, "parent_track_id" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetParentStepId, "parent_step_id" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetPID, "pid" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetMass, "mass" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetCharge, "charge" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetTime, "time" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetLength, "length" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetPosition, "position" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetMomentum, "momentum" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetVelocity, "velocity" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetSpeed, "speed" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetLorentzFactor, "lorentz_factor" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetKineticEnergy, "kinetic_energy" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetKineticEnergy_eV, "kinetic_energy_ev" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetPolarAngleToZ, "polar_angle_to_z" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetAzimuthalAngleToX, "azimuthal_angle_to_x" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetMagneticField, "magnetic_field" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetElectricField, "electric_field" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetMagneticGradient, "magnetic_gradient" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetElectricPotential, "electric_potential" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetLongMomentum, "long_momentum" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetTransMomentum, "trans_momentum" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetLongVelocity, "long_velocity" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetTransVelocity, "trans_velocity" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetLongMomentum, "long_momentum" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetPolarAngleToB, "polar_angle_to_b" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetCyclotronFrequency, "cyclotron_frequency" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetOrbitalMagneticMoment, "orbital_magnetic_moment" ) +
    		KSDictionary< KSParticle >::AddComponent( &KSParticle::GetGuidingCenterPosition, "guiding_center_position" ) +
			KSDictionary< KSParticle >::AddComponent( &KSParticle::GetCurrentSpaceName, "current_space_name" ) +
			KSDictionary< KSParticle >::AddComponent( &KSParticle::GetCurrentSurfaceName, "current_surface_name" ) +
			KSDictionary< KSParticle >::AddComponent( &KSParticle::GetCurrentSideName, "current_side_name" );


    static const int sKSKThreeVectorDict =
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::GetX, "x" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::GetY, "y" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::GetZ, "z" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::Magnitude, "magnitude" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::MagnitudeSquared, "magnitude_squared" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::Perp, "perp" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::PerpSquared, "perp_squared" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::PolarAngle, "polar_angle" ) +
    		KSDictionary< KThreeVector >::AddComponent( &KThreeVector::AzimuthalAngle, "azimuthal_angle" );

} /* namespace Kassiopeia */

