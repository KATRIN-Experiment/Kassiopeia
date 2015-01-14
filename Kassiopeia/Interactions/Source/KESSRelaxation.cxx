#include <iostream>
#include "KSInteractionsMessage.h"
#include "KESSRelaxation.h"
#include "KSParticleFactory.h"
#include "KSParticle.h"
#include "KESSScatteringCalculator.h"

#include "KConst.h"
using katrin::KConst;

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{

	using namespace std;

	KESSRelaxation::KESSRelaxation() :
        fSiliconBandGap(0.)
	{
	}

    void KESSRelaxation::RelaxAtom( unsigned int vacantShell,
                                    const KSParticle& aFinalParticle,
                                    KSParticleQueue& aQueue )
	{

        intmsg_debug( "KESSRelaxation::RelaxAtom" << eom );

		//find ionized shell depending on energy
		//1 - K Shell 1839eV
		//21 - L1 148.7eV
		//22 - L2 99.2eV
		//23 - L3 99.2eV
		//3 - M(11.4eV)
		//4 - M(uniform 0-11.4eV)

		if (vacantShell == 1)
		{
			intmsg_debug( "Relax - K Shell" << eom );

            RelaxKShell( aFinalParticle,
                         aQueue );
		}
		else if (vacantShell == 21)
		{
			intmsg_debug( "Relax - L1 Shell" << eom );

            RelaxL1Shell( aFinalParticle,
                          aQueue );
		}
		else if (vacantShell == 22 || vacantShell == 23)
		{
			intmsg_debug( "Relax - L23 Shell" << eom );

            RelaxL23Shell( aFinalParticle,
                           aQueue );
		}
		else
		{
			intmsg(eError) << "KESSRelaxation::RelaxAtom" << ret << "Unknown Shell: " << vacantShell << eom;
		}
	}

    void KESSRelaxation::RelaxL23Shell( const KSParticle& aFinalParticle,
                                        KSParticleQueue& aQueue )
	{

        CreateMMVacancies( 23,
                           aFinalParticle,
                           aQueue );

	}

    void KESSRelaxation::RelaxL1Shell( const KSParticle& aFinalParticle,
                                       KSParticleQueue& aQueue )
	{

		intmsg_debug( "KESSRelaxation::RelaxL1Shell" << eom );

        double eL1 = this->GetBindingEnergy(21);
        double eL2 = this->GetBindingEnergy(22);

        double augerEnergy = 0.;
        double random = KRandom::GetInstance().Uniform();


		//Fluorescence is 9.77x10e-6;
		if (random < 0.975)
		{	//			49.5eV
			//			L23 vacancy

            double holeEnergy = this->GetBindingEnergy(3) * KRandom::GetInstance().Uniform();
			augerEnergy = eL1 - eL2 - holeEnergy - fSiliconBandGap;

            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );
            this->RelaxL23Shell( aFinalParticle,
                                 aQueue );

		}
		else if (random >= 0.975 && random < 1)
		{
			//			132eV
			//			M,M vacancy
            CreateMMVacancies( 21,
                               aFinalParticle,
                               aQueue );

		}
		else
		{
			intmsg(eError) << "KESSRelaxation::RelaxL1Shell" << ret << "L1 Shell undefined probability!" << eom;
		}
		return;
	}

    void KESSRelaxation::RelaxKShell( const KSParticle& aFinalParticle,
                                      KSParticleQueue& aQueue )
	{

		intmsg(eDebug) << "KESSRelaxation::RelaxKShell" << eom;

        double augerEnergy = 0.;

        double eK = this->GetBindingEnergy(1);
        double eL1 = this->GetBindingEnergy(21);
        double eL2 = this->GetBindingEnergy(22);

        double random = KRandom::GetInstance().Uniform();

		/*
		 * transition	prob		vacancy
		 * KL1L1			19.2		L1 L1
		 * KL1L2			38.8		L1 L2
		 * KL2L2			23.3		L2 L2
		 * KL1M			7.5		L1 M
		 * KL2M			10.4		L2 M
		 * KMM			0.8		M M
		 */

		random -= 0.192;
		if (random <= 0)
		{
			//KL1L1
			intmsg_debug( "KL1L1" << eom );
			augerEnergy = eK - eL1 - eL1 - fSiliconBandGap;
            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );
            RelaxL1Shell( aFinalParticle,
                          aQueue );
            RelaxL1Shell( aFinalParticle,
                          aQueue );
			return;
		}
		random -= 0.388;
		if (random <= 0)
		{
			//KL1L2
			intmsg_debug( "KL1L2" << eom );
			augerEnergy = eK - eL1 - eL2 - fSiliconBandGap;
            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );
            RelaxL1Shell( aFinalParticle,
                          aQueue );
            RelaxL23Shell( aFinalParticle,
                           aQueue );
			return;
		}
		random -= 0.233;
		if (random <= 0)
		{
			//KL2L2
			intmsg_debug( "KL2L2" << eom );
			augerEnergy = eK - eL2 - eL2 - fSiliconBandGap;
            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );
            RelaxL23Shell( aFinalParticle,
                           aQueue );
            RelaxL23Shell( aFinalParticle,
                           aQueue);
			return;
		}
		random -= 0.075;
		if (random <= 0)
		{
			//KL1M
			intmsg_debug( "KL1M" << eom );

            double holeEnergy = this->GetBindingEnergy(3) * KRandom::GetInstance().Uniform();
			//To allow for a further DAQ simulation:    this->AddChargeCarriers(0.,holeEnergy);
			augerEnergy = eK - eL1 - holeEnergy - fSiliconBandGap;

            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );

            RelaxL1Shell( aFinalParticle,
                          aQueue );
			return;
		}

		random -= 0.104;
		if (random <= 0)
		{
			//KL2M
			intmsg_debug( "KL23M" << eom );

            double holeEnergy = this->GetBindingEnergy(3) * KRandom::GetInstance().Uniform();
			//To allow for a further DAQ simulation:    this->AddChargeCarriers(0.,holeEnergy);
			augerEnergy = eK - eL2 - holeEnergy - fSiliconBandGap;

            CreateAugerElectron( augerEnergy,
                                 aFinalParticle,
                                 aQueue );

            this->RelaxL23Shell( aFinalParticle,
                                 aQueue );
			return;
		}
		random -= 0.008;
		if (random <= 0)
		{
			//KMM
            CreateMMVacancies( 1,
                               aFinalParticle,
                               aQueue );
			return;
		}
		else
		{
			intmsg(eError) << "KESSRelaxation::RelaxKShell" << ret << "Probability Problem!" << eom;
			return;
		}
	}

    void KESSRelaxation::CreateMMVacancies(unsigned int fromThisShell,
                                           const KSParticle& aFinalParticle,
                                           KSParticleQueue& aQueue )
	{

		intmsg_debug( "KESSRelaxation::CreateMMVacancies" << ret << "MMVacany from: " << fromThisShell << eom );

        double random11;
        double probability;
        double augerEnergy = 0.;

		do
		{
			random11 = -1 + 2 * KRandom::GetInstance().Uniform();
            probability = 1 - std::fabs(random11);

		}
		while (KRandom::GetInstance().Uniform() > probability);

		augerEnergy = this->GetBindingEnergy(fromThisShell) - (1 + random11) * this->GetBindingEnergy(3) - fSiliconBandGap;

		// uncomment if you want to track holes for a further DAQ Simulation
		/* Double_t hole1=(1 + random11) * this->GetBindingEnergy(3)*KRandom::GetInstance()->Uniform();
		 Double_t hole2=(1 + random11) * this->GetBindingEnergy(3)-hole1;
		 this->AddChargeCarriers(0.,hole1);
		 this->AddChargeCarriers(0.,hole2);*/


        this->CreateAugerElectron( augerEnergy,
                                   aFinalParticle,
                                   aQueue );
	}

    double KESSRelaxation::GetBindingEnergy(unsigned int ionizedShell)
	{
		intmsg_debug( "KESSRelaxation::GetBindingEnergy" << eom );

		//find ionized shell depending on energy
		//1 - K Shell 1839eV
		//21 - L1 148.7eV
		//22 - L2 99.2eV
		//23 - L3 99.2eV
		//3 - M(11.4eV)
		//4 - M(0-11.4)

		switch (ionizedShell)
		{
			case 1:
				return 1839.;
				break;

			case 21:
				return 148.7;
				break;

			case 22:
				return 99.2;
				break;

			case 23:
				return 99.2;
				break;

			case 3:

				return 11.4;
				break;

			case 4:
				return 11.4 * KRandom::GetInstance().Uniform();
				break;

			default:
				intmsg(eError) << "KESSRelaxation::GetBindingEnergy" << ret << "Unknown Ionized Shell!" << eom;
				return 0;
		}

	}

    void KESSRelaxation::CreateAugerElectron(const double& augerEnergy_eV,
                                             const KSParticle& aFinalParticle,
                                             KSParticleQueue& aQueue)
	{
            KSParticle* AugerElectron = KSParticleFactory::GetInstance()->Create(11);

            (*AugerElectron) = aFinalParticle;
            AugerElectron->SetKineticEnergy_eV(augerEnergy_eV);
            AugerElectron->SetPolarAngleToZ( 180*std::acos(1-2*KRandom::GetInstance().Uniform())/KConst::Pi() );
            AugerElectron->SetAzimuthalAngleToX( KRandom::GetInstance().Uniform() * 360 );
            AugerElectron->SetLabel( "KESSRelaxation" );

            aQueue.push_back(AugerElectron);
	}

}//end namespace kassiopeia
