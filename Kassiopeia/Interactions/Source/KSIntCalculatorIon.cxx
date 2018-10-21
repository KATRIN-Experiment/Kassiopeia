#include "KSIntCalculatorIon.h"
#include "KSIntCalculatorHydrogen.h"
#include "KSParticleFactory.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{
  //Cross section data for hydrogen ions + H2 --> electron, taken from:
  //TATSUO TABATA, TOSHIZO SHIRAI
  //ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+, H2+, H3+, H, H2, AND H− WITH HYDROGEN MOLECULES
  //Atomic Data and Nuclear Data Tables, Volume 76, Issue 1, 2000, Pages 1-25, ISSN 0092-640X
  //http://dx.doi.org/10.1006/adnd.2000.0835
  //(http://www.sciencedirect.com/science/article/pii/S0092640X00908350)

  KSIntCalculatorIon::KSIntCalculatorIon() :
    fGas( "H_2" ),
    E_Binding( KConst::BindingEnergy_H2() )

  {
  }
  KSIntCalculatorIon::KSIntCalculatorIon( const KSIntCalculatorIon& aCopy ) :
    KSComponent(),
    fGas( aCopy.fGas )
  {
  }
  KSIntCalculatorIon* KSIntCalculatorIon::Clone() const
  {
    return new KSIntCalculatorIon( *this );
  }
  KSIntCalculatorIon::~KSIntCalculatorIon()
  {
  }
  
  void KSIntCalculatorIon::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection )
  {
    int aParticleID = aParticle.GetPID();
    double aEnergy = aParticle.GetKineticEnergy_eV()/1000;  //put in keV

    //Ionization of H_2
    if (fGas.compare("H_2") == 0) {
      
      E_Binding = KConst::BindingEnergy_H2();
      
      //H+,D+,T+
      if (aParticleID==2212||aParticleID==99041||aParticleID==99071) {
	aCrossSection = Hplus_H2_crossSection(aEnergy);
      }
      
      //H2+,D2+,T2+
      else if (aParticleID==99012||aParticleID==99042||aParticleID==99072) {
	aCrossSection = H2plus_H2_crossSection(aEnergy);
      }
      
      //H3+,D3+,T3+
      else if (aParticleID==99013||aParticleID==99043||aParticleID==99073) {
	aCrossSection = H3plus_H2_crossSection(aEnergy);
      }
      
      //H-,D-,T-
      else if (aParticleID==99021||aParticleID==99051||aParticleID==99081) {
	aCrossSection = Hminus_H2_crossSection(aEnergy);
      }
      
      else {
	aCrossSection = 0.;
	return;
      }
      
    }
    
    //else if (fGas.compare("He") == 0) {

    //E_Binding = KConst::BindingEnergy_He;

    //Add cross sections for He...
    
    // }
    
    else {
      aCrossSection = 0.;
      return;
    }
    
    return;
  }
    
  void KSIntCalculatorIon::ExecuteInteraction( const KSParticle& anIncomingIon, KSParticle& anOutgoingIon, KSParticleQueue& aSecondaries)
  {
    // incoming primary ion
    double tIncomingIonEnergy = anIncomingIon.GetKineticEnergy_eV();
    KThreeVector tIncomingIonPosition = anIncomingIon.GetPosition();
    KThreeVector tIncomingIonMomentum = anIncomingIon.GetMomentum();
    double tIncomingIonMass = anIncomingIon.GetMass();
    
    // outgoing secondary electron
    //Create electron
    KSParticle* tSecondaryElectron = KSParticleFactory::GetInstance().StringCreate( "e-" );
    //Set position (same as initial particle)
    tSecondaryElectron->SetPosition( tIncomingIonPosition );
    //Set energy
    double tSecondaryElectronEnergy;
    CalculateSecondaryElectronEnergy(tIncomingIonMass,tIncomingIonEnergy,tSecondaryElectronEnergy);
    tSecondaryElectron->SetKineticEnergy_eV( tSecondaryElectronEnergy );
    
    //Set angle (isotropic). Should be improved with distribution from the literature
    double tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) )*180/KConst::Pi();
    double tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() )*180/KConst::Pi();
    tSecondaryElectron->SetPolarAngleToZ( tTheta );
    tSecondaryElectron->SetAzimuthalAngleToX( tPhi );

    tSecondaryElectron->SetLabel( GetName() );
    aSecondaries.push_back( tSecondaryElectron );

    // outgoing primary ion
    anOutgoingIon = anIncomingIon;
    //Energy loss only from secondary electron and binding energy
    anOutgoingIon.SetKineticEnergy_eV( tIncomingIonEnergy - tSecondaryElectronEnergy - E_Binding );

    //Assume no deflection (this should be improved)
    //tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) )*180/KConst::Pi();
    //tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() )*180/KConst::Pi();
    //anOutgoingIon.SetPolarAngleToZ( tTheta );
    //anOutgoingIon.SetAzimuthalAngleToX( tPhi );

    // outgoing secondary ion (H2+). Available by not yet in use...
    //Isotropic distribution
    /*KSParticle* tSecondaryIon = KSParticleFactory::GetInstance().StringCreate( "H_2^+" );
    tSecondaryIon->SetPosition( tIncomingIonPosition );
    tSecondaryIon->SetPolarAngleToZ( tTheta );
    tSecondaryIon->SetAzimuthalAngleToX( tPhi );
    tSecondaryIon->SetKineticEnergy_eV( 0 );
    tSecondaryIon->SetLabel( GetName() );
    aSecondaries.push_back( tSecondaryIon );*/
    
    return;
  }

  void KSIntCalculatorIon::CalculateSecondaryElectronEnergy( const double anIncomingIonMass, const double anIncomingIonEnergy,double& aSecondaryElectronEnergy)
  {
    double tSecondaryElectronEnergy;
    double tCrossSection = 0;
    double I = E_Binding;

    //Diff. cross section for maximum possible secondary electron energy
    double aSecondaryElectronMaxEnergy = anIncomingIonEnergy - I; //extremely unlikely/impossible for electron to have this high of an energy
    double sigma_max;
    CalculateEnergyDifferentialCrossSection( anIncomingIonMass,
					     anIncomingIonEnergy,
    					     aSecondaryElectronMaxEnergy,
    					     sigma_max
    					     );
    
    //Diff. cross section for minimum possible secondary electron energy
    double aSecondaryElectronMinEnergy = 0;
    double sigma_min;
    CalculateEnergyDifferentialCrossSection( anIncomingIonMass,
					     anIncomingIonEnergy,
					     aSecondaryElectronMinEnergy,
					     sigma_min
					     );

    //Rejection sampling
    while ( true )
      {
	//Randomly select a possible electron energy
	tSecondaryElectronEnergy = KRandom::GetInstance().Uniform( aSecondaryElectronMinEnergy,
								   aSecondaryElectronMaxEnergy,
								   false, true
								   );
	//Get the diff. cross section for this electron energy
	CalculateEnergyDifferentialCrossSection( anIncomingIonMass,
						 anIncomingIonEnergy,
						 tSecondaryElectronEnergy,
						 tCrossSection
						 );
	//Randomly select a diff. cross section
	double tRandom = KRandom::GetInstance().Uniform( 0.,
							 sigma_min,
							 false,true
							 );
	//Trying to optimize the random sampling
	//See https://am207.github.io/2017/wiki/rejectionsampling.html
	/*double tRandom = KRandom::GetInstance().Uniform( 0.,
							 2*( (sigma_max-sigma_min)*tSecondaryElectronEnergy/(anIncomingIonEnergy-I)+sigma_min),
							 false,true
							 );*/
	//If the random diff. cross section is less than the actual diff. cross section (i.e. it lies within the distribution), use the electron energy
	if ( tRandom < tCrossSection )            
	  break;            
      }

    aSecondaryElectronEnergy = tSecondaryElectronEnergy;
  }

  //Taken from
  //M. E. Rudd, Differential cross sections for secondary electron production by proton impact, Phys. Rev. A 38, 6129, 1 December 1988
  void KSIntCalculatorIon::CalculateEnergyDifferentialCrossSection( const double anIncomingIonMass,
								      const double anIncomingIonEnergy,
								      const double aSecondaryElectronEnergy,
								      double &aCrossSection
								      )
  { 
    //Assume ionization of H_2
    double A_1 = 0.80;
    double B_1 = 2.9;
    double C_1 = 0.86;
    double D_1 = 1.48;
    double E_1 = 7.0;
    double A_2 = 1.06;
    double B_2 = 4.2;
    double C_2 = 1.39;
    double D_2 = 0.48;
    double alpha = 0.87;
    double N = 2; //number of electrons in atomic subshell of target
    double I = KConst::BindingEnergy_H2(); //Binding energy of target atom
    
    //Ionization of He
    //if (fGas.compare("He") == 0) {
    //  Put in numbers from Rudd
    //}
    
    double S = 4*KConst::Pi()*KConst::BohrRadiusSquared()*N*pow(KConst::ERyd_eV()/I,2);
    
    double lambda = anIncomingIonMass/KConst::M_el_eV(); //ratio of incoming projectile mass and electron mass
    double T = anIncomingIonEnergy/lambda;
    double v = sqrt(T/I); //reduced initial velocity
    double w = aSecondaryElectronEnergy / I;
    aCrossSection = (S/I) * ( F_1(v,A_1,B_1,C_1,D_1,E_1) + F_2(v,A_2,B_2,C_2,D_2)*w ) * pow(1+w,-3) * pow(1+exp(alpha*(w-w_c(v,I))/v),-1);
  }
  
  //Cross sections from TABATA SHIRAI (page 8-9)
  double KSIntCalculatorIon::Hplus_H2_crossSection(double aEnergy) {
    //analytic expression #9 in TABATA SHIRAI (page 8-9)
    const double E_threshold = 2.0e-2; //keV
    const double a1 = 1.864e-4;
    const double a2 = 1.216;
    const double a3 = 5.31e1;
    const double a4 = 8.97e-1;
    const double E_min = 7.50e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy,E_threshold);
    double value = 0;
    if( (aEnergy > E_min)&&(aEnergy < E_max)) {
      value = sigma1(E1,a1,a2,a3,a4);
    }
    return value;
  }
  
  double KSIntCalculatorIon::H2plus_H2_crossSection(double aEnergy) {
    //analytic expression #17 in TABATA SHIRAI (page 8-9)
    const double E_threshold = 3.0e-2;
    const double a1 = 1.086e-3;
    const double a2 = 1.153;
    const double a3 = 1.24e+1;
    const double a4 = -4.44e-1;
    const double a5 = 5.96e+1;
    const double a6 = 1.0;
    const double E_min = 3.16e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy,E_threshold);
    double value = 0;
    if( (aEnergy > E_min)&&(aEnergy < E_max)) {
      value = sigma6(E1,a1,a2,a3,a4,a5,a6);
    }
    return value;
  }

  double KSIntCalculatorIon::H3plus_H2_crossSection(double aEnergy) {
    //analytic expression #24 in TABATA SHIRAI (page 8-9)
    const double E_threshold = 3.6e-2;
    const double a1 = 2.63e-3;
    const double a2 = 9.31e-1;
    const double a3 = 4.05e-1;
    const double a4 = 1.0;
    const double a5 = 1.26e+2;
    const double a6 = 2.13e+2;
    const double E_min = 7.50e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy,E_threshold);
    double value = 0;
    if( (aEnergy > E_min)&&(aEnergy < E_max)) {
      value = sigma2(E1,a1,a2,a3,a4,a5,a6);
    }
    return value;
  }

  double KSIntCalculatorIon::Hminus_H2_crossSection(double aEnergy) {
    //analytic expression #47 in TABATA SHIRAI (page 8-9)
    const double E_threshold = 2.25e-3;
    const double a1 = 4.19e-2;
    const double a2 = 1.89;
    const double a3 = 1.78e-1;
    const double a4 = -2.3e-1;
    const double a5 = 1.04;
    const double a6 = 8.7e-1;
    const double a7 = 1.65e+1;
    const double a8 = 1.088;
    const double a9 = 5.33e-3;
    const double a10 = 1.66e-1;
    const double E_min = 2.37e-3;
    const double E_max = 5.00e+1;
    double E1 = E_1(aEnergy,E_threshold);
    double value = 0;
    if( (aEnergy > E_min)&&(aEnergy < E_max)) {
      value = sigma11(E1,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);
    }
    return value;
  }

  
  //Functions used in differential cross sections from Rudd
  double KSIntCalculatorIon::F_1(double v,double A_1,double B_1,double C_1,double D_1,double E_1) {
    double value = L_1(v,C_1,D_1,E_1)+H_1(v,A_1,B_1);
    return value;
  }
  double KSIntCalculatorIon::F_2(double v,double A_2,double B_2,double C_2,double D_2) {
    double value = L_2(v,C_2,D_2)*H_2(v,A_2,B_2)/(L_2(v,C_2,D_2)+H_2(v,A_2,B_2));
    return value;
  }
  double KSIntCalculatorIon::H_1(double v,double A_1,double B_1) {
    double value = A_1*log(1+pow(v,2))/(pow(v,2)+B_1/pow(v,2));
    return value;
  }
  double KSIntCalculatorIon::H_2(double v,double A_2,double B_2) {
    double value = A_2/pow(v,2)+B_2/pow(v,4);
    return value;
  }
  double KSIntCalculatorIon::L_1(double v,double C_1,double D_1,double E_1) {
    double value = C_1*pow(v,D_1)/(1+E_1*pow(v,D_1+4));
    return value;
  }
  double KSIntCalculatorIon::L_2(double v,double C_2,double D_2) {
    double value = C_2*pow(v,D_2);
    return value;
  }
  double KSIntCalculatorIon::w_c(double v,double I) {
    double w_2 = KConst::ERyd_eV()/(4*I);
    double value = 4*pow(v,2)-2*v-w_2;
    return value;
  }
  
  //Functions used in cross sections from TABATA SHIRAI (page 3)
  double KSIntCalculatorIon::f1(double x,double c1,double c2) {
    double ERyd_keV = KConst::ERyd_eV()/1000;
    double sigma0 = 1e-20; //m^2
    double value = sigma0*c1*pow((x/ERyd_keV),c2);
    return value;
  }

  double KSIntCalculatorIon::f2(double x,double c1,double c2,double c3,double c4) {
    double value = f1(x,c1,c2)/(1+pow((x/c3),c2+c4));
    return value;
  }

  double KSIntCalculatorIon::f3(double x,double c1,double c2,double c3,double c4,double c5,double c6) {
    double value = f1(x,c1,c2)/(1+pow((x/c3),c2+c4)+pow((x/c5),c2+c6));
    return value;
  }

  double KSIntCalculatorIon::sigma1(double E1,double a1,double a2,double a3,double a4) {
    double value = f2(E1,a1,a2,a3,a4);
    return value;
  }

  double KSIntCalculatorIon::sigma2(double E1,double a1,double a2,double a3,double a4,double a5,double a6) {
    double value = f2(E1,a1,a2,a3,a4)+a5*f2(E1/a6,a1,a2,a3,a4);
    return value;
  }

  double KSIntCalculatorIon::sigma6(double E1,double a1,double a2,double a3,double a4,double a5,double a6) {
    double value = f3(E1,a1,a2,a3,a4,a5,a6);
    return value;
  }

  double KSIntCalculatorIon::sigma10(double E1,double a1,double a2,double a3,double a4,double a5,double a6,double a7,double a8) {
    double value = f3(E1,a1,a2,a3,a4,a5,a6)+a7*f3(E1/a8,a1,a2,a3,a4,a5,a6);
    return value;
  }
  
  double KSIntCalculatorIon::sigma11(double E1,double a1,double a2,double a3,double a4,double a5,double a6,double a7,double a8,double a9,double a10) {
    double value = f3(E1,a1,a2,a3,a4,a5,a6)+f2(E1,a7,a8,a9,a10);
    return value;
  }

  //E is given in keV
  double KSIntCalculatorIon::E_1(double E,double E_threshold) {
    double value = E-E_threshold;
    return value;
  }
  
}

