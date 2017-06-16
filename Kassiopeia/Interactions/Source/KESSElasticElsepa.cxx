#include "KESSElasticElsepa.h"
#include "KSParticle.h"
#include <map>
#include <algorithm>

#include "KConst.h"
#include "KRandom.h"

#include "KSInteractionsMessage.h"

using namespace std;
using namespace katrin;

namespace Kassiopeia
{

    KESSElasticElsepa::KESSElasticElsepa()
    {
        fInteraction = string( "Elastic" );
        ReadMFP( "Elsepa_MFP.txt", fElScMFPMap );
        ReadPDF( "Elsepa.txt", fElScMap );
    }

    KESSElasticElsepa::KESSElasticElsepa (const KESSElasticElsepa &aCopy ):
        KSComponent(),
        fElScMFPMap( aCopy.fElScMFPMap ),
        fElScMap( aCopy.fElScMap )
    {
        fInteraction = aCopy.fInteraction;
    }

    KESSElasticElsepa* KESSElasticElsepa::Clone() const
    {
        return new KESSElasticElsepa( *this );
    }

    KESSElasticElsepa::~KESSElasticElsepa()
    {
    }

    void KESSElasticElsepa::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection)
    {
        double tKineticEnergy = aParticle.GetKineticEnergy_eV();

        //map iterator to the maps above and below electronEnergy_eV
        std::map< double, double >::iterator mapAbove, mapBelow;

        //find dictionary entry - mapAbove is the energy area ABOVE electronEnergy_eV
        mapAbove = fElScMFPMap.lower_bound( tKineticEnergy );

        //mapBelow the energy area BELOW electronEnergy_eV
        mapBelow = mapAbove;
        mapBelow--;

        double elMFP1 = mapBelow->second;
        double elMFP2 = mapAbove->second;

        double elENE1 = mapBelow->first;
        double elENE2 = mapAbove->first;

        //linear interpolation
        double MeanFreePathAngstroem = this->InterpolateLinear( tKineticEnergy,
                                                                elENE1,
                                                                elENE2,
                                                                elMFP1,
                                                                elMFP2 );

        //aCrossSection = 12.06 * 1E-6 / (MeanFreePathAngstroem * 1E-10 * KConst::N_A());
        // Molar Volume of Silicon = 12.06 * 1E-6 m^3/mol
        aCrossSection = 12.06 / ( MeanFreePathAngstroem * 1E-4 * KConst::N_A() );
    }

    void KESSElasticElsepa::ExecuteInteraction( const KSParticle& anInitialParticle,
                                                KSParticle& aFinalParticle,
                                                KSParticleQueue& /*aQueue*/ )
    {
        //Elastic scattering doesn't change the Energy
        double tKineticEnergy_eV = anInitialParticle.GetKineticEnergy_eV();

        //computes the new polar angle due to scattering
        double tTheta = GetScatteringPolarAngle( tKineticEnergy_eV );
        intmsg_debug( "KESSElasticElsepa::Execute" << ret
                      << "Elastic Calculator computed Theta =  " << 180*tTheta/KConst::Pi() <<eom );

        //dice an azimuthal angle
        double tPhi = 2. * KConst::Pi() * KRandom::GetInstance().Uniform();

        intmsg_debug( "KESSElasticElsepa::Execute" << ret
                      << "Randomly chosen azimuthal angle: " << 180*tPhi/KConst::Pi() <<eom );

        //update the final particle
        KThreeVector tInitialDirection = anInitialParticle.GetMomentum();
        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() *
                ( sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit())
                  + cos( tTheta ) * tInitialDirection.Unit()
                );

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tFinalDirection );
        aFinalParticle.SetKineticEnergy_eV( tKineticEnergy_eV );
        aFinalParticle.SetLabel( GetName() );


        fStepNInteractions = 1;
        fStepEnergyLoss = 0.;
    }

    double KESSElasticElsepa::GetScatteringPolarAngle( const double& aKineticEnergy )
    {

        double scatteringAngle_pi = 0;
        double rand2 = KRandom::GetInstance().Uniform();

        //map iterator to the maps above and below the kinetic energy of the electron
        std::map< double, std::vector< std::vector< double > > >::iterator mapAbove, mapBelow;

        //vector iterator for integral (probability)
        std::vector< double >::iterator intBelow, intAbove;

        //value (energy loss)
        double valBelow, valAbove;

        //find dictionary entry - mapAbove is the energy area ABOVE the kinetic energy of the electron
        mapAbove = fElScMap.lower_bound( aKineticEnergy );

        //mapBelow the energy area BELOW the kinetic energy of the electron
        mapBelow = mapAbove;
        mapBelow--;

        //search for the values above and below the probability integral
        intAbove = mapAbove->second.at( 1 ).end();
        intBelow = mapAbove->second.at( 1 ).begin();
        intAbove = std::lower_bound( intBelow, intAbove, rand2 );

        intBelow = intAbove;
        intBelow--;

        //fill the variables with the according energy losses
        valAbove = mapAbove->second.at( 0 ).at( intAbove - mapAbove->second.at( 1 ).begin() );
        valBelow = mapAbove->second.at( 0 ).at( intBelow - mapAbove->second.at( 1 ).begin() );

        //interpolate to actual random number and save
        double resultAbove = this->InterpolateLinear( rand2, *intBelow, *intAbove, valBelow, valAbove );

        //see above (but now for the lower energy area
        intAbove = mapBelow->second.at( 1 ).end();
        intBelow = mapBelow->second.at( 1 ).begin();

        intAbove = std::lower_bound( intBelow, intAbove, rand2 );

        intBelow = intAbove;
        intBelow--;
        valAbove = mapBelow->second.at( 0 ).at( intAbove - mapBelow->second.at( 1 ).begin() );
        valBelow = mapBelow->second.at( 0 ).at( intBelow - mapBelow->second.at( 1 ).begin() );

        double resultBelow = this->InterpolateLinear( rand2, *intBelow, *intAbove, valBelow, valAbove );

        //interpolate between the results for energy areas above and below the kinetic energy of the electron
        scatteringAngle_pi = this->InterpolateLinear( aKineticEnergy,
                                                      mapBelow->first,
                                                      mapAbove->first,
                                                      resultBelow,
                                                      resultAbove );

        return scatteringAngle_pi;
    }

}
