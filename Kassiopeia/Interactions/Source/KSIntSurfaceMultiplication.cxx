#include "KSIntSurfaceMultiplication.h"

#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

#include <iostream>
#include <cmath>
#include <limits>

namespace Kassiopeia
{

    KSIntSurfaceMultiplication::KSIntSurfaceMultiplication() :
            KSComponent(),
            fPerformSideCheck(false),
            fSideSignIsNegative(false),
            fSideName(std::string("both")),
            fEnergyLossFraction( 0. ),
            fEnergyRequiredPerParticle( std::numeric_limits<double>::max() )
    {
    }

    KSIntSurfaceMultiplication::KSIntSurfaceMultiplication( const KSIntSurfaceMultiplication& aCopy ) :
            KSComponent(),
            fPerformSideCheck(aCopy.fPerformSideCheck),
            fSideSignIsNegative(aCopy.fSideSignIsNegative),
            fSideName(aCopy.fSideName),
            fEnergyLossFraction( aCopy.fEnergyLossFraction ),
            fEnergyRequiredPerParticle( aCopy.fEnergyRequiredPerParticle )
    {
    }

    KSIntSurfaceMultiplication* KSIntSurfaceMultiplication::Clone() const
    {
        return new KSIntSurfaceMultiplication( *this );
    }

    KSIntSurfaceMultiplication::~KSIntSurfaceMultiplication()
    {
    }

    void KSIntSurfaceMultiplication::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue )
    {
        //determine the amount of energy we have to work with
        double tKineticEnergy = anInitialParticle.GetKineticEnergy();
        tKineticEnergy *= (1.0 - fEnergyLossFraction);

        //prevent kinetic energy from going negative
        if(tKineticEnergy < 0.0)
        {
            intmsg( eError ) << "surface diffuse interaction named <" << GetName() << "> tried to give a particle a negative kinetic energy." << eom;
            return;
        }

        //now determine the number of particles we will generate
        double tMean = tKineticEnergy/fEnergyRequiredPerParticle;
        unsigned int tNParticles = KRandom::GetInstance().Poisson(tMean);

        //energy should be probably be randomly partitioned, not completely equally distributed
        double tChildEnergy = tKineticEnergy/((double)(tNParticles));

        //figure out the basis directions for the particle ejections
        //we eject them with a diffuse 'Lambertian' distribution
        KThreeVector tNormal;
        if( anInitialParticle.GetCurrentSurface() != NULL )
        {
            tNormal = anInitialParticle.GetCurrentSurface()->Normal( anInitialParticle.GetPosition() );
        }
        else if( anInitialParticle.GetCurrentSide() != NULL )
        {
            tNormal = anInitialParticle.GetCurrentSide()->Normal( anInitialParticle.GetPosition() );
        }
        else
        {
            intmsg( eError ) << "surface diffuse interaction named <" << GetName() << "> was given a particle with neither a surface nor a side set" << eom;
            return;
        }

        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector momDirection = tInitialMomentum.Unit();

        double dot_prod = tInitialMomentum.Dot( tNormal );
        KThreeVector tInitialNormalMomentum =  dot_prod*tNormal;

        tInitialNormalMomentum = -1.0*tInitialNormalMomentum; //reverse direction for reflection
        KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;

        tInitialNormalMomentum = tInitialNormalMomentum.Unit();
        tInitialTangentMomentum = tInitialTangentMomentum.Unit();
        KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross( tInitialNormalMomentum.Unit() );

        bool execute_interaction = true;
        if(fPerformSideCheck)
        {

            if(fSideSignIsNegative && dot_prod > 0)
            {
                execute_interaction = false;
            }
            if( !fSideSignIsNegative && dot_prod < 0 )
            {
                execute_interaction = false;
            }
        }

        if(execute_interaction) //only execute interaction if the specified side of this surface is active
        {
            //now generate the ejected particles
            for(unsigned int i=0; i<tNParticles; i++)
            {
                KSParticle* tParticle = new KSParticle( anInitialParticle );

                //dice direction
                double tAzimuthalAngle = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );
                double tSinTheta = KRandom::GetInstance().Uniform( 0., 0.99 );
                double tCosTheta = std::sqrt( (1.0 - tSinTheta)*(1.0 + tSinTheta) );

                KThreeVector tDirection;
                tDirection = tCosTheta*tInitialNormalMomentum;
                tDirection += tSinTheta*std::cos(tAzimuthalAngle)*tInitialTangentMomentum.Unit();
                tDirection += tSinTheta*std::sin(tAzimuthalAngle)*tInitialOrthogonalMomentum.Unit();

                if(tDirection.Dot(momDirection) > 0)
                {
                    tDirection = -1.0*tDirection;
                }

                tParticle->SetMomentum(tDirection);
                tParticle->SetKineticEnergy(tChildEnergy);
                tParticle->SetCurrentSurface(NULL);
                aQueue.push_back(tParticle);
            }
        }

        //kill parent
        aFinalParticle = anInitialParticle;
        aFinalParticle.SetActive(false);
        aFinalParticle.AddLabel(GetName());
        aFinalParticle.SetMomentum(-1.0*tInitialMomentum);
        aFinalParticle.SetKineticEnergy(0);

        return;
    }

}
