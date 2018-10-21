#include "KSTermMagnetron.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

    KSTermMagnetron::KSTermMagnetron() :
        fMaxPhi( 360. ),
        fFirstStep( true ),
        fPhiFirstStep( 0. ),
        fPositionBefore( 0., 0., 0. ),
        fAtanJump( false ),
        fJumpDirection( 0 )
    {
    }
    KSTermMagnetron::KSTermMagnetron( const KSTermMagnetron& aCopy ) :
        KSComponent(),
        fMaxPhi( aCopy.fMaxPhi ),
        fFirstStep( aCopy.fFirstStep ),
        fPhiFirstStep( aCopy.fPhiFirstStep ),
        fPositionBefore( aCopy.fPositionBefore ),
        fAtanJump( aCopy.fAtanJump ),
        fJumpDirection( aCopy.fJumpDirection )
    {
    }
    KSTermMagnetron* KSTermMagnetron::Clone() const
    {
        return new KSTermMagnetron( *this );
    }
    KSTermMagnetron::~KSTermMagnetron()
    {
    }
    
    void KSTermMagnetron::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        double phiLastStep;
        double phiThisStep;
        double DeltaPhi = 0.;
        
        if( fFirstStep == true )
        {
            DeltaPhi = 0.;
            fPositionBefore = anInitialParticle.GetPosition();
            fPhiFirstStep = KConst::Pi() + atan2( anInitialParticle.GetPosition().Y(), anInitialParticle.GetPosition().X() );
            fFirstStep = false;
        }
        else
        {
            phiLastStep = KConst::Pi() + atan2( fPositionBefore.Y(), fPositionBefore.X() );
            phiThisStep = KConst::Pi() + atan2( anInitialParticle.GetPosition().Y(), anInitialParticle.GetPosition().X() );
            
            if( fAtanJump == false )
            {
                DeltaPhi = fabs( fPhiFirstStep - phiThisStep );
            }
            
            if( fAtanJump == true && fJumpDirection == 1 )
            {
                DeltaPhi = 2 * KConst::Pi() + phiThisStep - fPhiFirstStep;
            }
            
            if( fAtanJump == true && fJumpDirection == 2 )
            {
                DeltaPhi = 2 * KConst::Pi() - phiThisStep + fPhiFirstStep;
            }
            
            if( phiLastStep - phiThisStep >= KConst::Pi() )
            {
                fAtanJump = true;
                fJumpDirection = 1;
                DeltaPhi = 2 * KConst::Pi() + phiThisStep - fPhiFirstStep;
            }
            
            if( phiLastStep - phiThisStep <= -KConst::Pi() )
            {
                fAtanJump = true;
                fJumpDirection = 2;
                DeltaPhi = 2 * KConst::Pi() - phiThisStep + fPhiFirstStep;
            }
            
            fPositionBefore = anInitialParticle.GetPosition();
        }
        
        if( DeltaPhi > fMaxPhi * KConst::Pi() / 180.)
        {
            fAtanJump = false;
            fJumpDirection = 0;

            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMagnetron::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
