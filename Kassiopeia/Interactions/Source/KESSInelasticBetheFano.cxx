#include "KESSInelasticBetheFano.h"
#include "KSParticle.h"
#include "KConst.h"
#include "KRandom.h"
using katrin::KRandom;
#include <map>
#include "KSInteractionsMessage.h"
#include "KESSPhotoAbsorbtion.h"
#include "KESSRelaxation.h"
#include <algorithm>

namespace Kassiopeia
{

    KESSInelasticBetheFano::KESSInelasticBetheFano() :
        fBetheFanoDepositedEnergy( 0. ),
        fRho( 2.33e-24 )
    {
        fInteraction = string( "Inelastic" );
        ReadMFP( "BetheFano_MFP.txt", fInElScMFPMap );
        ReadPDF( "BetheFano.txt", fInElScMap );
    }

    KESSInelasticBetheFano::KESSInelasticBetheFano( const KESSInelasticBetheFano &aCopy ):
        KSComponent(),
        fBetheFanoDepositedEnergy( aCopy.fBetheFanoDepositedEnergy ),
        fInElScMFPMap( aCopy.fInElScMFPMap ),
        fInElScMap( aCopy.fInElScMap ),
        fRho( aCopy.fRho )
    {
    }

    KESSInelasticBetheFano* KESSInelasticBetheFano::Clone() const
    {
        return new KESSInelasticBetheFano( *this );
    }

    KESSInelasticBetheFano::~KESSInelasticBetheFano()
    {
    }

    void KESSInelasticBetheFano::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection )
    {
        double tKineticEnergy = aParticle.GetKineticEnergy_eV();

        //map iterator to the maps above and below electronEnergy_eV
        std::map< double, double >::iterator mapAbove, mapBelow;

        //find dictionary entry - mapAbove is the energy area ABOVE electronEnergy_eV
        mapAbove = fInElScMFPMap.lower_bound( tKineticEnergy );

        //mapBelow the energy area BELOW electronEnergy_eV
        mapBelow = mapAbove;
        mapBelow--;

        double inElMFP1 = mapBelow->second;
        double inElMFP2 = mapAbove->second;

        double inElENE1 = mapBelow->first;
        double inElENE2 = mapAbove->first;

        double help = this->InterpolateLinear( tKineticEnergy, inElENE1, inElENE2, inElMFP1, inElMFP2 );

        //for BetheFano it is not the MeanFreePath that is stored, but a crossection-like variable
        double MeanFreePathAngstroem = KConst::M_Si() / (KConst::N_A() * fRho * help);

        //aCrossSection = 12.06 * 1E-6 / (MeanFreePathAngstroem * 1E-10 * KConst::N_A());
        aCrossSection = 12.06 / (MeanFreePathAngstroem * 1E-4 * KConst::N_A());
    }

    void KESSInelasticBetheFano::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue )
    {
        //reset deposited energy counter.
        fBetheFanoDepositedEnergy = 0.;

        double tKineticEnergy_eV = anInitialParticle.GetKineticEnergy_eV();
        intmsg_debug( "KESSInelasticBetheFano::Execute" << ret
                << "Kinetic energy of particle: " << tKineticEnergy_eV << eom );

        //computes the energy loss due to scattering
        double tEloss_eV = CalculateEnergyLoss( tKineticEnergy_eV );
        intmsg_debug( "KESSInelasticBetheFano::Execute" << ret
                << "Inelastic Calculator computed ELoss =  " << tEloss_eV << eom );

        //computes the new angle due to scattering
        double tTheta = CalculateScatteringAngle( tEloss_eV, tKineticEnergy_eV );
        intmsg_debug( "KESSInelasticBetheFano::Execute" << ret
                << "Inelastic Calculator computed Azimutal Scattering Angle Theta =  "
                << 180*tTheta/KConst::Pi() << eom );

        //dice an azimuthal angle
        double tPhi = 2. * KConst::Pi() * KRandom::GetInstance().Uniform();
        intmsg_debug( "KESSInelasticBetheFano::Execute" << ret
                << "Randomly chosen azimuthal angle: " << 180*tPhi/KConst::Pi() << eom );

        //check if the new energy is too low
        if( tKineticEnergy_eV - tEloss_eV < KConst::Q() * 0.1 )
        {
            tKineticEnergy_eV = tEloss_eV + KConst::Q() * 0.1;

            intmsg_debug( "KESSInelasticBetheFano::Execute" << ret
                    << "Warning: Energy after elastic scattering too low. Energy artificially increased. "<<eom );
        }

        //update the final particle
        KThreeVector tInitialDirection = anInitialParticle.GetMomentum();
        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() * (sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit()) + cos( tTheta ) * tInitialDirection.Unit());

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tFinalDirection );
        aFinalParticle.SetKineticEnergy_eV( tKineticEnergy_eV - tEloss_eV );
        aFinalParticle.SetLabel( GetName() );

        fStepEnergyLoss = tEloss_eV;
        fStepNInteractions = 1;

        unsigned int IonisedShell = 0;
        if( fIonisationCalculator )
        {
            //!\todo check what is the energy point of reference
            IonisedShell = fIonisationCalculator->IonizeShell( tEloss_eV, aFinalParticle, aQueue );
        }
        else
        {
            //deposit energy locally
            fBetheFanoDepositedEnergy = tEloss_eV;
        }

        if( fRelaxationCalculator )
        {
            //doublecheck ionization has happened
            if( fIonisationCalculator )
            {
                if( IonisedShell != 3 )
                {
                    //!\todo check what is the energy point of reference
                    fRelaxationCalculator->RelaxAtom( IonisedShell, aFinalParticle, aQueue );
                }
            }
            else
            {
                intmsg( eError ) << "KESSInelasticBetheFano::Execute" << ret << "Trying to do AugerCascade without DeltaRay Production (Ionisation)" << eom;
            }
        }
        else
        {
            //deposit energy locally
            if( fIonisationCalculator )
            {
                if( IonisedShell != 3 )
                {
                    fBetheFanoDepositedEnergy += fIonisationCalculator->GetBindingEnergy( IonisedShell );
                }
            }
        }

    }

    double KESSInelasticBetheFano::CalculateEnergyLoss( const double& EKin )
    {
        if( EKin > 107.0 && EKin < 409000.0 )
        {
            double energyLoss = 0.;
            double resultAbove = 0.;
            double rand3 = KRandom::GetInstance().Uniform();

            //map iterator to the maps above and below the kinetic energy of the electron
            std::map< double, std::vector< std::vector< double > > >::iterator mapAbove, mapBelow;

            //vector iterator for integral (probability)
            std::vector< double >::iterator intBelow, intAbove;

            //value (energy loss)
            double valBelow, valAbove;

            //find dictionary entry - mapAbove is the energy area ABOVE kinetic energy of the electron
            mapAbove = fInElScMap.lower_bound( EKin );

            //mapBelow the energy area BELOW kinetic energy of the electron
            mapBelow = mapAbove;
            mapBelow--;

            //search for the values above and below the probability integral
            intAbove = mapAbove->second.at( 1 ).end();
            intBelow = mapAbove->second.at( 1 ).begin();

            //the integral doesn't start at 0 all the time. get new random number if no value is found
            while( rand3 <= *intBelow )
            {
                rand3 = KRandom::GetInstance().Uniform();
            }

            intAbove = lower_bound( intBelow, intAbove, rand3 );

            intBelow = intAbove;
            intBelow--;

            //fill the variables with the according energy losses
            valAbove = mapAbove->second.at( 0 ).at( intAbove - mapAbove->second.at( 1 ).begin() );
            valBelow = mapAbove->second.at( 0 ).at( intBelow - mapAbove->second.at( 1 ).begin() );

            //interpolate to actual random number and save
            resultAbove = this->InterpolateLinear( rand3, *intBelow, *intAbove, valBelow, valAbove );

            //see above (but now for the lower energy area)
            intAbove = mapBelow->second.at( 1 ).end();
            intBelow = mapBelow->second.at( 1 ).begin();

            intAbove = lower_bound( intBelow, intAbove, rand3 );

            intBelow = intAbove;
            intBelow--;

            valAbove = mapBelow->second.at( 0 ).at( intAbove - mapBelow->second.at( 1 ).begin() );
            valBelow = mapBelow->second.at( 0 ).at( intBelow - mapBelow->second.at( 1 ).begin() );

            double resultBelow = this->InterpolateLinear( rand3, *intBelow, *intAbove, valBelow, valAbove );

            //interpolate between the results for energy areas above and below the kinetic energy of the electron
            energyLoss = this->InterpolateLinear( EKin, mapBelow->first, mapAbove->first, resultBelow, resultAbove );

            return energyLoss;
        }
        else
        {
            intmsg( eError ) << "Energy of particle out of range for PDF/MFP tables." << " Please check your Terminators to ensure that the particle energy" << " is always between 107 < E < 409,000 eV when using BetheFano!" << eom;
            return 0.0;
        }

    }

    double KESSInelasticBetheFano::CalculateScatteringAngle( const double& EnergyLoss, const double& KineticEnergy )
    {
        return std::asin( std::sqrt( EnergyLoss / KineticEnergy ) );
    }

}
