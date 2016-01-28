#include "KSTrajTermPropagation.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSTrajTermPropagation::KSTrajTermPropagation() :
            fDirection( eForward )
    {
    }
    KSTrajTermPropagation::KSTrajTermPropagation( const KSTrajTermPropagation& aCopy ) :
            KSComponent(),
            fDirection( aCopy.fDirection )
    {
    }
    KSTrajTermPropagation* KSTrajTermPropagation::Clone() const
    {
        return new KSTrajTermPropagation( *this );
    }
    KSTrajTermPropagation::~KSTrajTermPropagation()
    {
    }

    void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle, KSTrajExactDerivative& aDerivative ) const
    {
        KThreeVector tVelocity = fDirection * aParticle.GetVelocity();
        KThreeVector tForce = aParticle.GetCharge() * (aParticle.GetElectricField() + tVelocity.Cross( aParticle.GetMagneticField() ));

        aDerivative.AddToVelocity( tVelocity );
        aDerivative.AddToForce( tForce );

        return;
    }

    void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const
    {
        double tLongVelocity = fDirection * aParticle.GetLongVelocity();
        double tLongitudinalMomentum = aParticle.GetLongMomentum();
        double tTransverseMomentum = aParticle.GetTransMomentum();
        double tLorentzFactor = aParticle.GetLorentzFactor();
        double tOrbitalMagneticMoment = aParticle.GetOrbitalMagneticMoment();

        KThreeVector tElectricfield = aParticle.GetElectricField();
        KThreeVector tMagneticField = aParticle.GetMagneticField();
        KThreeMatrix tMagneticGradient = aParticle.GetMagneticGradient();
        KThreeVector tMagneticFieldUnit = tMagneticField.Unit();
        double tMagneticFieldMagnitude = tMagneticField.Magnitude();
        double tMagneticGradientUnit = tMagneticFieldUnit * (tMagneticGradient * tMagneticFieldUnit);

        double tLongitudinalForce = -1. * (tOrbitalMagneticMoment / tLorentzFactor) * tMagneticGradientUnit + aParticle.GetCharge() * tElectricfield.Dot( tMagneticField.Unit() );
        double tTransverseForce = ((tLongitudinalMomentum * tTransverseMomentum) / (2 * aParticle.GetMass() * tLorentzFactor * tMagneticFieldMagnitude)) * tMagneticGradientUnit;

        trajmsg_debug( "adiabatic propagation gc velocity: <" << tLongVelocity * tMagneticFieldUnit << ">" << ret )
        trajmsg_debug( "adiabatic propagation longitudinal force <" << tLongitudinalForce << ">" << ret )
        trajmsg_debug( "adiabatic propagation transverse force <" << tTransverseForce << ">" << ret )

        aDerivative.AddToGuidingCenterVelocity( tLongVelocity * tMagneticFieldUnit );
        aDerivative.AddToLongitudinalForce( tLongitudinalForce );
        aDerivative.AddToTransverseForce( tTransverseForce );

        return;
    }

    void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aParticle, KSTrajMagneticDerivative& aDerivative ) const
    {
    	KThreeVector tVelocity = fDirection * aParticle.GetMagneticField().Unit();

        aDerivative.AddToVelocity( tVelocity );

        return;
    }

    void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajElectricParticle& aParticle, KSTrajElectricDerivative& aDerivative ) const
    {
    	KThreeVector tVelocity = fDirection * aParticle.GetElectricField().Unit();

        aDerivative.AddToVelocity( tVelocity );

        return;
    }

    void KSTrajTermPropagation::SetDirection( const Direction& aDirection )
    {
        fDirection = aDirection;
        return;
    }

}
