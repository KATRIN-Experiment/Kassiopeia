#include "KSTrajControlTime.h"

namespace Kassiopeia
{

    KSTrajControlTime::KSTrajControlTime() :
            fTime( 0. )
    {
    }
    KSTrajControlTime::KSTrajControlTime( const KSTrajControlTime& aCopy ) :
            KSComponent(),
            fTime( aCopy.fTime )
    {
    }
    KSTrajControlTime* KSTrajControlTime::Clone() const
    {
        return new KSTrajControlTime( *this );
    }
    KSTrajControlTime::~KSTrajControlTime()
    {
    }

    void KSTrajControlTime::Calculate( const KSTrajExactParticle&, double& aValue )
    {
        aValue = fTime;
        return;
    }
    void KSTrajControlTime::Check( const KSTrajExactParticle&, const KSTrajExactParticle&, const KSTrajExactError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlTime::Calculate( const KSTrajAdiabaticParticle&, double& aValue )
    {
        aValue = fTime;
        return;
    }
    void KSTrajControlTime::Check( const KSTrajAdiabaticParticle&, const KSTrajAdiabaticParticle&, const KSTrajAdiabaticError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlTime::Calculate( const KSTrajElectricParticle&, double& aValue )
    {
        aValue = fTime;
        return;
    }
    void KSTrajControlTime::Check( const KSTrajElectricParticle&, const KSTrajElectricParticle&, const KSTrajElectricError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

    void KSTrajControlTime::Calculate( const KSTrajMagneticParticle&, double& aValue )
    {
        aValue = fTime;
        return;
    }
    void KSTrajControlTime::Check( const KSTrajMagneticParticle&, const KSTrajMagneticParticle&, const KSTrajMagneticError&, bool& aFlag )
    {
        aFlag = true;
        return;
    }

}
