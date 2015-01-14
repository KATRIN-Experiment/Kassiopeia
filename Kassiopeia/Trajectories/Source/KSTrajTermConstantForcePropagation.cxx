#include "KSTrajTermConstantForcePropagation.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSTrajTermConstantForcePropagation::KSTrajTermConstantForcePropagation()
    {
    }
    KSTrajTermConstantForcePropagation::KSTrajTermConstantForcePropagation( const KSTrajTermConstantForcePropagation& )
    {
    }
    KSTrajTermConstantForcePropagation* KSTrajTermConstantForcePropagation::Clone() const
    {
        return new KSTrajTermConstantForcePropagation( *this );
    }    
    KSTrajTermConstantForcePropagation::~KSTrajTermConstantForcePropagation()
    {
    }

    void KSTrajTermConstantForcePropagation::Differentiate( const KSTrajExactParticle& aParticle, KSTrajExactDerivative& aDerivative ) const
    {
        KThreeVector tVelocity = aParticle.GetVelocity();

        aDerivative.AddToVelocity( tVelocity );
        aDerivative.AddToForce( fForce );

        return;
    }

    void KSTrajTermConstantForcePropagation::SetForce( const KThreeVector& aForce )
    {
        fForce = aForce;
    }
}
