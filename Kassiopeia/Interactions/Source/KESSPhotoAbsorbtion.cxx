#include <iostream>
#include "KConst.h"
#include "KESSPhotoAbsorbtion.h"
#include "KSParticle.h"
#include "KSParticleFactory.h"
#include "KRandom.h"
using katrin::KRandom;
#include "KFile.h"
#include "KSInteractionsMessage.h"
#include <map>
#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{
    using namespace std;

    KESSPhotoAbsorbtion::KESSPhotoAbsorbtion():
                fSiliconBandGap( 0. ),
                fPhotoDepositedEnergy( 0. ),
                fShellL1( 0. ),
                fShellL2( 0. ),
                fShellL3( 0. ),
                fShellM( 0. )
    {
        intmsg_debug( "KESSPhotoAbsorption::KESSPhotoAbsorption" << eom );

        this->ReadIonisationPDF( "PhotoAbsorbtion.txt" );
    }

    KESSPhotoAbsorbtion::~KESSPhotoAbsorbtion()
    {
    }

    void KESSPhotoAbsorbtion::ReadIonisationPDF( std::string data_filename )
    {
        intmsg_debug( "KESSPhotoAbsorption::ReadIonisationPDF" << eom );

        char line[ 196 ];
        double one = 0, two = 0, three = 0, four = 0, five = 0;

        std::string myPathToTable = DATA_DEFAULT_DIR;

        std::string UnusedReturnValue;

        FILE *ionizationFile = fopen( (myPathToTable + "/" + data_filename).c_str(), "r" );

        if( ionizationFile == 0 )
        {
            intmsg( eError ) << "KESSPhotoAbsorbtion::ReadIonisationPDF"
                             << ret << "FILE " << myPathToTable + "/" + data_filename
                             << " NOT FOUND CHECK IN IO CONIGFILE FOR DATA DIRECTORY" << eom;
        }

        UnusedReturnValue = fgets( line, 195, ionizationFile );

        unsigned int lineN = 0;

        while( fgets( line, 195, ionizationFile ) != NULL )
        {
            sscanf( line, "%lf %lf %lf %lf %lf", &one, &two, &three, &four, &five );

            if( feof( ionizationFile ) == false )
            {
                fIonizationMap.insert( std::pair< double, unsigned int >( one, lineN ) );
                fShellL1.push_back( two );
                fShellL2.push_back( three );
                fShellL3.push_back( four );
                fShellM.push_back( five );
                lineN++;
            }
        }
        fclose( ionizationFile );
        return;
    }

    double KESSPhotoAbsorbtion::GetBindingEnergy( unsigned int ionizedShell )
    {
        //find ionized shell depending on energy
        //1 - K Shell 1839eV
        //21 - L1 148.7eV
        //22 - L2 99.2eV
        //23 - L3 99.2eV
        //3 - M(11.4eV)

        switch( ionizedShell ) {
            case 1 :
                return 1839.;
                break;

            case 21 :
                return 148.7;
                break;

            case 22 :
                return 99.2;
                break;

            case 23 :
                return 99.2;
                break;

            case 3 :

                return 11.4;
                break;

            default :

                intmsg( eError ) << "KESSPhotoAbsorbtion::GetBindingEnergy" << ret << "Unknown Ionized Shell!" << eom;
                return 0;
        }

    }

    void KESSPhotoAbsorbtion::CreateSecondary( const double secondaryEnergy,
                                               const KSParticle& aFinalParticle,
                                               KSParticleQueue& aQueue )
    {
        //Dapor 2009 - 	arXiv:0903.4805v1 [cond-mat.mtrl-sci]
        //Shimizu 1992 - Rep. Prog. Phys. (1992) 487-531
        double tPhi = KRandom::GetInstance().Uniform( 0.0, 2 * KConst::Pi() );
        double tTheta = std::acos( 1 - 2 * KRandom::GetInstance().Uniform() );

        KThreeVector tInitialDirection = aFinalParticle.GetMomentum();
        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() *
                    ( sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit())
                      + cos( tTheta ) * tInitialDirection.Unit()
                    );

        KSParticle* tSecondary = KSParticleFactory::GetInstance()->Create(11);
        (*tSecondary) = aFinalParticle;
        tSecondary->SetMomentum( tFinalDirection );
        tSecondary->SetKineticEnergy_eV( secondaryEnergy );
        tSecondary->SetLabel( "KESSPhotoaAbsorbtion" );
        aQueue.push_back( tSecondary );

        return;
    }

    unsigned int KESSPhotoAbsorbtion::IonizeShell( const double& lostEnergy_eV,
                                                   const KSParticle& aFinalParticle,
                                                   KSParticleQueue& aQueue )
    {        
        intmsg_debug( "KESSPhotoAbsorbtion::IonizeShell" << eom );

        fPhotoDepositedEnergy = 0.;
        double secondaryEnergy = 0.;
        double holeEnergy = 0.;
        double depositedEnergy = 0.;

        //find ionized shell depending on energy
        //1 - K Shell 1839eV
        //21 - L1 148.7eV
        //22 - L2 99.2eV
        //23 - L3 99.2eV
        //3 - M(11.4eV)
        double eK = this->GetBindingEnergy( 1 );
        //double eL1 = this->GetBindingEnergy(21);	FH: Why excluded???
        double eL2 = this->GetBindingEnergy( 22 );
        double eM = this->GetBindingEnergy( 3 );

        unsigned int ionizedShell = 0;

        //energy transfer lies above K edge
        if( lostEnergy_eV > eK )
        {
            //92% for K, 8% for L1
            if( KRandom::GetInstance().Uniform() < 0.92 )
            {
                ionizedShell = 1;
            }
            else
            {
                ionizedShell = 21;
            }

            //energy transfer between L23 and K ->use probabilities
        }
        else if( lostEnergy_eV <= eK && lostEnergy_eV > eL2 )
        {
            ionizedShell = FindIonizedShell( lostEnergy_eV );

            //*****************************
            //* M SHELL
            //*****************************
            if( ionizedShell == 3 )
            {
                holeEnergy = this->GetBindingEnergy( 3 ) * KRandom::GetInstance().Uniform();

                depositedEnergy += holeEnergy;
                secondaryEnergy = lostEnergy_eV - holeEnergy - fSiliconBandGap;
                this->CreateSecondary( secondaryEnergy,
                                       aFinalParticle,
                                       aQueue);
            }

            //energy transfer L23 -> M Shell ionization
        }
        else if( lostEnergy_eV <= eL2 && lostEnergy_eV > eM )
        {
            ionizedShell = 3;
            holeEnergy = this->GetBindingEnergy( 3 ) * KRandom::GetInstance().Uniform();
            depositedEnergy += holeEnergy;
            secondaryEnergy = lostEnergy_eV - holeEnergy - fSiliconBandGap;
            this->CreateSecondary( secondaryEnergy,
                                   aFinalParticle,
                                   aQueue);

        }
        else if( lostEnergy_eV <= eM )
        {
            ionizedShell = 3;
            holeEnergy = (lostEnergy_eV - fSiliconBandGap) * KRandom::GetInstance().Uniform();
            depositedEnergy += holeEnergy;
            secondaryEnergy = lostEnergy_eV - holeEnergy - fSiliconBandGap;
            this->CreateSecondary( secondaryEnergy,
                                   aFinalParticle,
                                   aQueue);
        }
        else
        {
            intmsg( eError ) << "KESSPhotoAbsorbtion::IonizeShell" << ret << "Shell Ionisation Error!" << eom;
            ionizedShell = 3;
        }

        //K,L1,L23 shell secondaries
        if( ionizedShell != 3 )
        {
            secondaryEnergy = lostEnergy_eV - this->GetBindingEnergy( ionizedShell ) - fSiliconBandGap;
            this->CreateSecondary( secondaryEnergy,
                                   aFinalParticle,
                                   aQueue);
        }
        else if( ionizedShell == 3 )
        {
            //do nothing
        }
        else
        {
            intmsg( eError ) << "KESSPhotoAbsorbtion::IonizeShell" << ret << "Unknown Shell: " << ionizedShell << eom;
        }

        fPhotoDepositedEnergy += depositedEnergy;

        return ionizedShell;
    }

    unsigned int KESSPhotoAbsorbtion::FindIonizedShell( double lostEnergy_eV )
    {
        double randomNumber = KRandom::GetInstance().Uniform();

        //find ionized shell depending on energy
        //1 - K Shell 1839eV
        //21 - L1 148.7eV
        //22 - L2 99.2eV
        //23 - L3 99.2eV
        //3 - M(11.4eV)

        //map iterator to the value below lostEnergy_eV
        std::map< double, unsigned int >::iterator mapBelow;
        mapBelow = fIonizationMap.upper_bound( lostEnergy_eV );
        mapBelow--;
        randomNumber -= fShellM.at( mapBelow->second );
        if( randomNumber <= 0 )
            return 3;
        randomNumber -= fShellL1.at( mapBelow->second );
        if( randomNumber <= 0 )
            return 21;
        randomNumber -= fShellL2.at( mapBelow->second );
        if( randomNumber <= 0 )
            return 22;
        randomNumber -= fShellL3.at( mapBelow->second );
        if( randomNumber <= 0 )
            return 23;

        intmsg( eWarning ) << "KESSPhotoAbsorbtion::FindIonizedShell" << ret
                           << "Probability mismatch in KESSPhotoAbsorbtion at energy: "
                           << lostEnergy_eV << eom;
        return 3;
    }

}
//end namespace kassiopeia
