#include "KSTrajControlLength.h"

namespace Kassiopeia
{

    KSTrajControlLength::KSTrajControlLength() :
            fLength( 0.0 )
    {
    }
    KSTrajControlLength::KSTrajControlLength( const KSTrajControlLength& aCopy ) :
            KSComponent(),
            fLength( aCopy.fLength )
    {
    }
    KSTrajControlLength* KSTrajControlLength::Clone() const
    {
        return new KSTrajControlLength( *this );
    }
    KSTrajControlLength::~KSTrajControlLength()
    {
    }

    void KSTrajControlLength::Calculate( const KSTrajExactParticle& aParticle, double& aValue )
    {
        double tLongVelocity = aParticle.GetLongVelocity();
        double tTransVelocity = aParticle.GetTransVelocity();
        double tSpeed = sqrt( tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity );
        aValue = fLength / tSpeed;
        return;
    }
    void KSTrajControlLength::Check( const KSTrajExactParticle&, const KSTrajExactParticle&, const KSTrajExactError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlLength::Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue )
    {
        double tLongVelocity = aParticle.GetLongVelocity();
        double tTransVelocity = aParticle.GetTransVelocity();
        double tSpeed = sqrt( tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity );
        aValue = fLength / tSpeed;
        return;
    }
    void KSTrajControlLength::Check( const KSTrajAdiabaticParticle&, const KSTrajAdiabaticParticle&, const KSTrajAdiabaticError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlLength::Calculate( const KSTrajMagneticParticle& aParticle, double& aValue )
    {
        double tLongVelocity = aParticle.GetLongVelocity();
        double tTransVelocity = aParticle.GetTransVelocity();
        double tSpeed = sqrt( tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity );
        aValue = fLength / tSpeed;
        return;
    }
    void KSTrajControlLength::Check( const KSTrajMagneticParticle&, const KSTrajMagneticParticle&, const KSTrajMagneticError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlLength::Calculate( const KSTrajElectricParticle& aParticle, double& aValue )
    {
        double tLongVelocity = aParticle.GetLongVelocity();
        double tTransVelocity = aParticle.GetTransVelocity();
        double tSpeed = sqrt( tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity );
        aValue = fLength / tSpeed;
        return;
    }
    void KSTrajControlLength::Check( const KSTrajElectricParticle&, const KSTrajElectricParticle&, const KSTrajElectricError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

}
