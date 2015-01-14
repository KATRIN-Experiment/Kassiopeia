#include "KSTrajTermSynchrotron.h"

#include "KConst.h"

namespace Kassiopeia
{

    KSTrajTermSynchrotron::KSTrajTermSynchrotron() :
            fEnhancement( 1. )
    {
    }
    KSTrajTermSynchrotron::KSTrajTermSynchrotron( const KSTrajTermSynchrotron& aCopy ) :
            fEnhancement( aCopy.fEnhancement )
    {
    }
    KSTrajTermSynchrotron* KSTrajTermSynchrotron::Clone() const
    {
        return new KSTrajTermSynchrotron( *this );
    }
    KSTrajTermSynchrotron::~KSTrajTermSynchrotron()
    {
    }

    void KSTrajTermSynchrotron::Differentiate( const KSTrajExactParticle& aParticle, KSTrajExactDerivative& aDerivative ) const
    {
        double Q = aParticle.GetCharge();
        double M = aParticle.GetMass();
        double P = aParticle.GetMomentum().Magnitude();
        double Gamma = aParticle.GetLorentzFactor();
        double Factor = (KConst::MuNull() / (6. * KConst::Pi() * KConst::C())) * ((Q * Q * Q * Q) / (M * P * P));

        KThreeVector tTUnit = aParticle.GetMomentum().Unit();
        KThreeVector tUUnit = aParticle.GetMomentum().Cross( aParticle.GetMagneticField() ).Unit();
        KThreeVector tVUnit = tTUnit.Cross( tUUnit ).Unit();

        double tET = aParticle.GetElectricField().Dot( tTUnit );
        double tEU = aParticle.GetElectricField().Dot( tUUnit );
        double tEV = aParticle.GetElectricField().Dot( tVUnit );
        double tBV = aParticle.GetMagneticField().Dot( tVUnit );
        double tXi1 = tET * tET + tBV * tBV * (P / M) * (P / M);
        double tXi2 = -2. * tEU * tBV * (P / M);
        double tXi3 = tEU * tEU + tEV * tEV;

        KThreeVector tForce = -fEnhancement * Factor * (Gamma * tXi1 + Gamma * Gamma * tXi2 + Gamma * Gamma * Gamma * tXi3) * aParticle.GetMomentum();

        aDerivative.AddToForce( tForce );

        return;
    }
    void KSTrajTermSynchrotron::Differentiate( const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const
    {
        double Q = aParticle.GetCharge();
        double M = aParticle.GetMass();
        double Factor = (KConst::MuNull() / (6. * KConst::Pi() * KConst::C())) * ((Q * Q * Q * Q) / (M * M * M));

        double tForce = -fEnhancement * Factor * aParticle.GetLorentzFactor() * aParticle.GetMagneticField().MagnitudeSquared() * aParticle.GetTransMomentum();

        aDerivative.AddToTransverseForce( tForce );

        return;
    }

    void KSTrajTermSynchrotron::SetEnhancement( const double& anEnhancement )
    {
        fEnhancement = anEnhancement;
        return;
    }

}
