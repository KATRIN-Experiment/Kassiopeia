#include "KSIntCalculatorHydrogen.h"
#include "KSInteractionsMessage.h"
#include "KSParticleFactory.h"

#include "KTextFile.h"
using katrin::KTextFile;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KConst.h"
using katrin::KConst;

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
    /////////////////////////////////////
    /////		Elastic	Base		/////
    /////////////////////////////////////

    KSIntCalculatorHydrogenElasticBase::KSIntCalculatorHydrogenElasticBase()
    {
    }

    KSIntCalculatorHydrogenElasticBase::~KSIntCalculatorHydrogenElasticBase()
    {
    }

    void KSIntCalculatorHydrogenElasticBase::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection )
    {
        CalculateCrossSection( aParticle.GetKineticEnergy_eV(), aCrossSection );
        return;
    }

    void KSIntCalculatorHydrogenElasticBase::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
        KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

        // outgoing primary

        double tLostKineticEnergy;
        double tTheta;
        double tPhi;

        CalculateTheta( tInitialKineticEnergy, tTheta );
        CalculateEloss( tInitialKineticEnergy, tTheta, tLostKineticEnergy );

        tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );

        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() * (sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit()) + cos( tTheta ) * tInitialDirection.Unit());

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetTime( anInitialParticle.GetTime() );
        aFinalParticle.SetPosition( anInitialParticle.GetPosition() );
        aFinalParticle.SetMomentum( tFinalDirection );
        aFinalParticle.SetKineticEnergy_eV( tInitialKineticEnergy - tLostKineticEnergy );
        aFinalParticle.AddLabel( GetName() );

        return;
    }

    void KSIntCalculatorHydrogenElasticBase::CalculateTheta( const double anEnergy, double& aTheta )
    {
        double clight = 1. / KConst::Alpha();
        double T, c, b, G, a, gam, K2, Gmax;

        double tDiffCrossSection;
        double tRandom;

        if( anEnergy >= 250. )
            Gmax = 1.e-19;
        else if( anEnergy < 250. && anEnergy >= 150. )
            Gmax = 2.5e-19;
        else
            Gmax = 1.e-18;

        T = anEnergy / 27.2;
        gam = 1. + T / (clight * clight);
        b = 2. / (1. + gam) / T;
        for( int i = 1; i < 5000; i++ )
        {
            tRandom = KRandom::GetInstance().Uniform( 0.0, 1.0, false, true );
            c = 1. + b - b * (2. + b) / (b + 2. * tRandom);
            K2 = 2. * T * (1. + gam) * fabs( 1. - c );
            a = (4. + K2) * (4. + K2) / (gam * gam);
            CalculateDifferentialCrossSection( anEnergy, c, tDiffCrossSection );
            G = a * tDiffCrossSection;
            tRandom = KRandom::GetInstance().Uniform( 0.0, 1.0, false, true );
            if( G > Gmax * tRandom )
                break;
        }
        aTheta = acos( c );

    }

    void KSIntCalculatorHydrogenElasticBase::CalculateDifferentialCrossSection( const double anEnergy, const double cosTheta, double& aCrossSection )
    {
        // Nishimura et al., J. Phys. Soc. Jpn. 54 (1985) 1757.

        double a02 = KConst::BohrRadiusSquared();
        double clight = 1. / KConst::Alpha(); // velocity of light in atomic units is 1/ alpha

        double Cel[ 50 ] =
        { -0.512, -0.512, -0.509, -0.505, -0.499, -0.491, -0.476, -0.473, -0.462, -0.452, -0.438, -0.422, -0.406, -0.388, -0.370, -0.352, -0.333, -0.314, -0.296, -0.277, -0.258, -0.239, -0.221, -0.202, -0.185, -0.167, -0.151, -0.135, -0.120, -0.105, -0.092, -0.070, -0.053, -0.039, -0.030, -0.024, -0.019, -0.016, -0.014, -0.013, -0.012, -0.009, -0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.002, -0.001 };

        double e[ 10 ] =
        { 0., 3., 6., 12., 20., 32., 55., 85., 150., 250. };

        double t[ 10 ] =
        { 0., 10., 20., 30., 40., 60., 80., 100., 140., 180. };

        double D[ 9 ][ 10 ] =
        {
        { 2.9, 2.70, 2.5, 2.10, 1.80, 1.2000, 0.900, 1.0000, 1.600, 1.9 },
        { 4.2, 3.60, 3.1, 2.50, 1.90, 1.1000, 0.800, 0.9000, 1.300, 1.4 },
        { 6.0, 4.40, 3.2, 2.30, 1.80, 1.1000, 0.700, 0.5400, 0.500, 0.6 },
        { 6.0, 4.10, 2.8, 1.90, 1.30, 0.6000, 0.300, 0.1700, 0.160, 0.23 },
        { 4.9, 3.20, 2.0, 1.20, 0.80, 0.3000, 0.150, 0.0900, 0.050, 0.05 },
        { 5.2, 2.50, 1.2, 0.64, 0.36, 0.1300, 0.050, 0.0300, 0.016, 0.02 },
        { 4.0, 1.70, 0.7, 0.30, 0.16, 0.0500, 0.020, 0.0130, 0.010, 0.01 },
        { 2.8, 1.10, 0.4, 0.15, 0.07, 0.0200, 0.010, 0.0070, 0.004, 0.003 },
        { 1.2, 0.53, 0.2, 0.08, 0.03, 0.0074, 0.003, 0.0016, 0.001, 0.0008 } };

        double T, K2, K, d, st1, st2, DH, gam, CelK, Ki, theta;
        aCrossSection = -1.0;
        int i, j;
        T = anEnergy / 27.2;
        if( anEnergy >= 250. )
        {
            gam = 1. + T / (clight * clight); // relativistic correction factor
            K2 = 2. * T * (1. + gam) * (1. - cosTheta);
            if( K2 < 0. )
                K2 = 1.e-30;
            K = sqrt( K2 );
            if( K < 1.e-9 )
                K = 1.e-9; // momentum transfer
            d = 1.4009; // distance of protons in H2
            st1 = 8. + K2;
            st2 = 4. + K2;
            // DH is the diff. cross section for elastic electron scatt.
            // on atomic hydrogen within the first Born approximation :
            DH = 4. * st1 * st1 / (st2 * st2 * st2 * st2) * a02;
            // CelK calculation with linear interpolation.
            // CelK is the correction of the elastic electron
            // scatt. on molecular hydrogen compared to the independent atom
            // model.
            if( K < 3. )
            {
                i = (int) (K / 0.1); //WOLF int->double->int
                Ki = i * 0.1;
                CelK = Cel[ i ] + (K - Ki) / 0.1 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else if( K >= 3. && K < 5. )
            {
                i = (int) (30 + (K - 3.) / 0.2); //WOLF: int->double
                Ki = 3. + (i - 30) * 0.2; //WOLF: int->double
                CelK = Cel[ i ] + (K - Ki) / 0.2 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else if( K >= 5. && K < 9.49 )
            {
                i = (int) (40 + (K - 5.) / 0.5); //WOLF: int->double
                Ki = 5. + (i - 40) * 0.5; //WOLF: int->double
                CelK = Cel[ i ] + (K - Ki) / 0.5 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else
                CelK = 0.;

            aCrossSection = 2. * gam * gam * DH * (1. + sin( K * d ) / (K * d)) * (1. + CelK);
        } //end if anE>=250
        else
        {
            theta = acos( cosTheta ) * 180. / KConst::Pi();
            for( i = 0; i <= 8; i++ )
                if( anEnergy >= e[ i ] && anEnergy < e[ i + 1 ] )
                    for( j = 0; j <= 8; j++ )
                        if( theta >= t[ j ] && theta < t[ j + 1 ] )
                            aCrossSection = 1.e-20 * (D[ i ][ j ] + (D[ i ][ j + 1 ] - D[ i ][ j ]) * (theta - t[ j ]) / (t[ j + 1 ] - t[ j ]));
        }

        return;
    }

    /////////////////////////////////
    /////		Elastic			/////
    /////////////////////////////////

    KSIntCalculatorHydrogenElastic::KSIntCalculatorHydrogenElastic()
    {
    }

    KSIntCalculatorHydrogenElastic::KSIntCalculatorHydrogenElastic( const KSIntCalculatorHydrogenElastic& )
    {
    }

    KSIntCalculatorHydrogenElastic* KSIntCalculatorHydrogenElastic::Clone() const
    {
        return new KSIntCalculatorHydrogenElastic( *this );
    }

    KSIntCalculatorHydrogenElastic::~KSIntCalculatorHydrogenElastic()
    {
    }

    void KSIntCalculatorHydrogenElastic::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
//        See: Liu, Phys. Rev. A35 (1987) 591,
//        Trajmar, Phys Reports 97 (1983) 221.

        const double e[ 14 ] =
        { 0., 1.5, 5., 7., 10., 15., 20., 30., 60., 100., 150., 200., 300., 400. };
        const double s[ 14 ] =
        { 9.6, 13., 15., 12., 10., 7., 5.6, 3.3, 1.1, 0.9, 0.5, 0.36, 0.23, 0.15 };

        const double emass = 1. / (KConst::Alpha() * KConst::Alpha());
        const double a02 = KConst::BohrRadiusSquared();

        double gam, T;
        T = anEnergie / 27.2;
        if( anEnergie >= 400. )
        {
            gam = (emass + T) / emass;
            aCrossSection = gam * gam * KConst::Pi() / (2. * T) * (4.2106 - 1. / T) * a02;
        }
        else
        {
            for( unsigned int i = 0; i <= 12; i++ )
            {
                if( anEnergie >= e[ i ] && anEnergie < e[ i + 1 ] )
                    aCrossSection = 1.e-20 * (s[ i ] + (s[ i + 1 ] - s[ i ]) * (anEnergie - e[ i ]) / (e[ i + 1 ] - e[ i ]));
            }
        }

        return;
    }

    void KSIntCalculatorHydrogenElastic::CalculateEloss( const double anEnergie, const double aTheta, double& anEloss )
    {
        double H2molmass = 69.e6;
        double emass = 1. / (KConst::Alpha() * KConst::Alpha());
        double cosTheta = cos( aTheta );

        anEloss = 2. * emass / H2molmass * (1. - cosTheta) * anEnergie;

        //check if electron won energy by elastic scattering on a molecule; this keeps electron energies around the gas temperature
        if( anEnergie < 1. )
        {
            double rndNr = KRandom::GetInstance().Uniform();
            double rndAngle = KRandom::GetInstance().Uniform();

            //generation of molecule velocity by maxwell-boltzmann distribution
            double Gx = sqrt( -2. * log( rndNr ) ) * cos( 2. * KConst::Pi() * rndAngle );
            double Gy = sqrt( -2. * log( rndNr ) ) * sin( 2. * KConst::Pi() * rndAngle );
            double Gz = sqrt( -2. * log( KRandom::GetInstance().Uniform() ) ) * cos( 2. * KConst::Pi() * KRandom::GetInstance().Uniform() );

            //thermal velocity of gas molecules
            double T = 300.; //gas temperature
            double sigmaT = sqrt( KConst::kB() * T / (2. * KConst::M_prot_kg()) );
            KThreeVector MolVelocity( sigmaT * Gx, sigmaT * Gy, sigmaT * Gz );

            //new electron velocity vector and energy:

            //assume electron velocity along z
            KThreeVector ElVelocity( 0., 0., sqrt( 2. * anEnergie * KConst::Q() / KConst::M_el_kg() ) );
            //relative velocity electron-molecule
            KThreeVector RelativeVelocity = ElVelocity - MolVelocity;
            //transformation into CMS
            KThreeVector CMSVelocity = (KConst::M_el_kg() / (KConst::M_el_kg() + KConst::M_prot_kg()) * ElVelocity + 2. * KConst::M_prot_kg() / (KConst::M_el_kg() + KConst::M_prot_kg()) * MolVelocity);
            //generation of random direction
            KThreeVector Random( KRandom::GetInstance().Uniform(), KRandom::GetInstance().Uniform(), KRandom::GetInstance().Uniform() );
            //new electron velocity
            ElVelocity = KConst::M_prot_kg() / (KConst::M_prot_kg() + KConst::M_el_kg()) * RelativeVelocity.Magnitude() * Random + CMSVelocity;
            anEloss = anEnergie - KConst::M_el_kg() / (2. * KConst::Q()) * ElVelocity.Magnitude() * ElVelocity.Magnitude();
        }
        return;
    }

    /////////////////////////////////
    /////		Vibration		/////
    /////////////////////////////////

    KSIntCalculatorHydrogenVib::KSIntCalculatorHydrogenVib()
    {
    }

    KSIntCalculatorHydrogenVib::KSIntCalculatorHydrogenVib( const KSIntCalculatorHydrogenVib& )
    {
    }

    KSIntCalculatorHydrogenVib* KSIntCalculatorHydrogenVib::Clone() const
    {
        return new KSIntCalculatorHydrogenVib( *this );
    }

    KSIntCalculatorHydrogenVib::~KSIntCalculatorHydrogenVib()
    {
    }

    void KSIntCalculatorHydrogenVib::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
        unsigned int i;

        static double sigma1[ 8 ] =
        { 0.0, 0.006, 0.016, 0.027, 0.033, 0.045, 0.057, 0.065 };

        static double sigma2[ 9 ] =
        { 0.065, 0.16, 0.30, 0.36, 0.44, 0.47, 0.44, 0.39, 0.34 };

        static double sigma3[ 7 ] =
        { 0.34, 0.27, 0.21, 0.15, 0.12, 0.08, 0.07 };

        if( anEnergie <= 0.5 || anEnergie > 10. )
        {
            aCrossSection = 0.;
        }
        else
        {
            if( anEnergie >= 0.5 && anEnergie < 1.0 )
            {
                i = (anEnergie - 0.5) / 0.1;
                aCrossSection = 1.e-20 * (sigma1[ i ] + (sigma1[ i + 1 ] - sigma1[ i ]) * (anEnergie - 0.5 - i * 0.1) * 10.);
            }
            else
            {
                if( anEnergie >= 1.0 && anEnergie < 5.0 )
                {
                    i = (anEnergie - 1.0) / 0.5;
                    aCrossSection = 1.e-20 * (sigma2[ i ] + (sigma2[ i + 1 ] - sigma2[ i ]) * (anEnergie - 1.0 - i * 0.5) * 2.);
                }
                else
                {
                    i = (anEnergie - 5.0) / 1.0;
                    aCrossSection = 1.e-20 * (sigma3[ i ] + (sigma3[ i + 1 ] - sigma3[ i ]) * (anEnergie - 5.0 - i * 1.0));
                }
            }
        }
        return;
    }

    void KSIntCalculatorHydrogenVib::CalculateEloss( const double, const double, double& anEloss )
    {
        anEloss = 0.5;
    }

    /////////////////////////////////
    /////		Rot02			/////
    /////////////////////////////////

    KSIntCalculatorHydrogenRot02::KSIntCalculatorHydrogenRot02()
    {
    }

    KSIntCalculatorHydrogenRot02::KSIntCalculatorHydrogenRot02( const KSIntCalculatorHydrogenRot02& )
    {
    }

    KSIntCalculatorHydrogenRot02* KSIntCalculatorHydrogenRot02::Clone() const
    {
        return new KSIntCalculatorHydrogenRot02( *this );
    }

    KSIntCalculatorHydrogenRot02::~KSIntCalculatorHydrogenRot02()
    {
    }

    void KSIntCalculatorHydrogenRot02::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
        unsigned int i;

        static double sigma2[ 8 ] =
        { 0.065, 0.069, 0.073, 0.077, 0.081, 0.085, 0.088, 0.090 };

        static double sigma3[ 10 ] =
        { 0.09, 0.11, 0.15, 0.20, 0.26, 0.32, 0.39, 0.47, 0.55, 0.64 };

        static double sigma4[ 9 ] =
        { 0.64, 1.04, 1.37, 1.58, 1.70, 1.75, 1.76, 1.73, 1.69 };

        static double sigma5[ 7 ] =
        { 1.69, 1.58, 1.46, 1.35, 1.25, 1.16, 1.0 };

        static double DeltaE = 0.045;

        if( anEnergie <= DeltaE + 1.e-8 || anEnergie > 10. )
        {
            aCrossSection = 0.;
        }
        else
        {
            if( anEnergie >= 0.045 && anEnergie < 0.1 )
            {
                i = (anEnergie - 0.045) / 0.01;
                aCrossSection = 1.e-20 * (sigma2[ i ] + (sigma2[ i + 1 ] - sigma2[ i ]) * (anEnergie - 0.045 - i * 0.01) * 100.);
            }
            else
            {
                if( anEnergie >= 0.1 && anEnergie < 1.0 )
                {
                    i = (anEnergie - 0.1) / 0.1;
                    aCrossSection = 1.e-20 * (sigma3[ i ] + (sigma3[ i + 1 ] - sigma3[ i ]) * (anEnergie - 0.1 - i * 0.1) * 10.);
                }
                else
                {
                    if( anEnergie >= 1.0 && anEnergie < 5.0 )
                    {
                        i = (anEnergie - 1.0) / 0.5;
                        aCrossSection = 1.e-20 * (sigma4[ i ] + (sigma4[ i + 1 ] - sigma4[ i ]) * (anEnergie - 1.0 - i * 0.5) * 2.);
                    }
                    else
                    {
                        i = (anEnergie - 5.0) / 1.0;
                        aCrossSection = 1.e-20 * (sigma5[ i ] + (sigma5[ i + 1 ] - sigma5[ i ]) * (anEnergie - 5.0 - i * 1.0));
                    }
                }
            }
        }

        return;
    }

    void KSIntCalculatorHydrogenRot02::CalculateEloss( const double, const double, double& anEloss )
    {
        anEloss = 0.045;
    }

    /////////////////////////////////
    /////		Rot13			/////
    /////////////////////////////////

    KSIntCalculatorHydrogenRot13::KSIntCalculatorHydrogenRot13()
    {
    }

    KSIntCalculatorHydrogenRot13::KSIntCalculatorHydrogenRot13( const KSIntCalculatorHydrogenRot13& )
    {
    }

    KSIntCalculatorHydrogenRot13* KSIntCalculatorHydrogenRot13::Clone() const
    {
        return new KSIntCalculatorHydrogenRot13( *this );
    }

    KSIntCalculatorHydrogenRot13::~KSIntCalculatorHydrogenRot13()
    {
    }

    void KSIntCalculatorHydrogenRot13::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
        unsigned int i;

        static double sigma2[ 6 ] =
        { 0.035, 0.038, 0.041, 0.044, 0.047, 0.05 };

        static double sigma3[ 10 ] =
        { 0.05, 0.065, 0.09, 0.11, 0.14, 0.18, 0.21, 0.25, 0.29, 0.33 };

        static double sigma4[ 9 ] =
        { 0.33, 0.55, 0.79, 0.94, 1.01, 1.05, 1.05, 1.04, 1.01 };

        static double sigma5[ 7 ] =
        { 1.01, 0.95, 0.88, 0.81, 0.75, 0.69, 0.62 };

        static double DeltaE = 0.075;

        if( anEnergie <= DeltaE + 1.e-8 || anEnergie > 10. )
        {
            aCrossSection = 0.;
        }
        else
        {
            if( anEnergie >= 0.075 && anEnergie < 0.1 )
            {
                i = (anEnergie - 0.075) / 0.01;
                aCrossSection = 1.e-20 * (sigma2[ i ] + (sigma2[ i + 1 ] - sigma2[ i ]) * (anEnergie - 0.075 - i * 0.01) * 100.);
            }
            else
            {
                if( anEnergie >= 0.1 && anEnergie < 1.0 )
                {
                    i = (anEnergie - 0.1) / 0.1;
                    aCrossSection = 1.e-20 * (sigma3[ i ] + (sigma3[ i + 1 ] - sigma3[ i ]) * (anEnergie - 0.1 - i * 0.1) * 10.);
                }
                else
                {
                    if( anEnergie >= 1.0 && anEnergie < 5.0 )
                    {
                        i = (anEnergie - 1.0) / 0.5;
                        aCrossSection = 1.e-20 * (sigma4[ i ] + (sigma4[ i + 1 ] - sigma4[ i ]) * (anEnergie - 1.0 - i * 0.5) * 2.);
                    }
                    else
                    {
                        i = (anEnergie - 5.0) / 1.0;
                        aCrossSection = 1.e-20 * (sigma5[ i ] + (sigma5[ i + 1 ] - sigma5[ i ]) * (anEnergie - 5.0 - i * 1.0));
                    }
                }
            }
        }
        return;
    }

    void KSIntCalculatorHydrogenRot13::CalculateEloss( const double, const double, double& anEloss )
    {
        anEloss = 0.075;
    }

    /////////////////////////////////
    /////		Rot20			/////
    /////////////////////////////////

    KSIntCalculatorHydrogenRot20::KSIntCalculatorHydrogenRot20()
    {
    }

    KSIntCalculatorHydrogenRot20::KSIntCalculatorHydrogenRot20( const KSIntCalculatorHydrogenRot20& )
    {
    }

    KSIntCalculatorHydrogenRot20* KSIntCalculatorHydrogenRot20::Clone() const
    {
        return new KSIntCalculatorHydrogenRot20( *this );
    }

    KSIntCalculatorHydrogenRot20::~KSIntCalculatorHydrogenRot20()
    {
    }

    void KSIntCalculatorHydrogenRot20::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
        unsigned int i;

        double Ep = anEnergie + 0.045;

        static double sigma2[ 8 ] =
        { 0.065, 0.069, 0.073, 0.077, 0.081, 0.085, 0.088, 0.090 };

        static double sigma3[ 10 ] =
        { 0.09, 0.11, 0.15, 0.20, 0.26, 0.32, 0.39, 0.47, 0.55, 0.64 };

        static double sigma4[ 9 ] =
        { 0.64, 1.04, 1.37, 1.58, 1.70, 1.75, 1.76, 1.73, 1.69 };

        static double sigma5[ 7 ] =
        { 1.69, 1.58, 1.46, 1.35, 1.25, 1.16, 1.0 };

        static double DeltaE = 0.045;

        if( Ep <= DeltaE + 1.e-8 || Ep > 10. )
        {
            aCrossSection = 0.;
        }
        else
        {
            if( Ep >= 0.045 && Ep < 0.1 )
            {
                i = (Ep - 0.045) / 0.01;
                aCrossSection = 1.e-20 * (sigma2[ i ] + (sigma2[ i + 1 ] - sigma2[ i ]) * (Ep - 0.045 - i * 0.01) * 100.);
            }
            else
            {
                if( Ep >= 0.1 && Ep < 1.0 )
                {
                    i = (Ep - 0.1) / 0.1;
                    aCrossSection = 1.e-20 * (sigma3[ i ] + (sigma3[ i + 1 ] - sigma3[ i ]) * (Ep - 0.1 - i * 0.1) * 10.);
                }
                else
                {
                    if( Ep >= 1.0 && Ep < 5.0 )
                    {
                        i = (Ep - 1.0) / 0.5;
                        aCrossSection = 1.e-20 * (sigma4[ i ] + (sigma4[ i + 1 ] - sigma4[ i ]) * (Ep - 1.0 - i * 0.5) * 2.);
                    }
                    else
                    {
                        i = (Ep - 5.0) / 1.0;
                        aCrossSection = 1.e-20 * (sigma5[ i ] + (sigma5[ i + 1 ] - sigma5[ i ]) * (Ep - 5.0 - i * 1.0));
                    }
                }
            }
        }

        aCrossSection = 1. / 5. * Ep / anEnergie * aCrossSection;
    }

    void KSIntCalculatorHydrogenRot20::CalculateEloss( const double, const double, double& anEloss )
    {
        anEloss = -0.045;
    }

    /////////////////////////////////////
    /////		Excitation Base		/////
    /////////////////////////////////////

    KSIntCalculatorHydrogenExcitationBase::KSIntCalculatorHydrogenExcitationBase()
    {
    }

    KSIntCalculatorHydrogenExcitationBase::~KSIntCalculatorHydrogenExcitationBase()
    {
    }

    void KSIntCalculatorHydrogenExcitationBase::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection )
    {
        CalculateCrossSection( aParticle.GetKineticEnergy_eV(), aCrossSection );
        return;
    }

    void KSIntCalculatorHydrogenExcitationBase::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
        KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

        // outgoing primary

        double tLostKineticEnergy;
        double tTheta;
        double tPhi;

        CalculateTheta( tInitialKineticEnergy, tTheta );
        CalculateEloss( tInitialKineticEnergy, tTheta, tLostKineticEnergy );

        tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );

        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() * (sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit()) + cos( tTheta ) * tInitialDirection.Unit());

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetTime( anInitialParticle.GetTime() );
        aFinalParticle.SetPosition( anInitialParticle.GetPosition() );
        aFinalParticle.SetMomentum( tFinalDirection );
        aFinalParticle.SetKineticEnergy_eV( tInitialKineticEnergy - tLostKineticEnergy );
        aFinalParticle.AddLabel( GetName() );

        return;
    }

    void KSIntCalculatorHydrogenExcitationBase::CalculateTheta( const double anEnergy, double& aTheta )
    {
        double clight = 1. / KConst::Alpha();
        double T, c, b, G, a, gam, K2, Gmax;

        double tDiffCrossSection;
        double tRandom;

        if( anEnergy >= 250. )
            Gmax = 1.e-19;
        else if( anEnergy < 250. && anEnergy >= 150. )
            Gmax = 2.5e-19;
        else
            Gmax = 1.e-18;

        T = anEnergy / 27.2;
        gam = 1. + T / (clight * clight);
        b = 2. / (1. + gam) / T;
        for( int i = 1; i < 5000; i++ )
        {
            tRandom = KRandom::GetInstance().Uniform( 0.0, 1.0, false, true );
            c = 1. + b - b * (2. + b) / (b + 2. * tRandom);
            K2 = 2. * T * (1. + gam) * fabs( 1. - c );
            a = (4. + K2) * (4. + K2) / (gam * gam);
            CalculateDifferentialCrossSection( anEnergy, c, tDiffCrossSection );
            G = a * tDiffCrossSection;
            tRandom = KRandom::GetInstance().Uniform( 0.0, 1.0, false, true );
            if( G > Gmax * tRandom )
                break;
        }
        aTheta = acos( c );

    }

    void KSIntCalculatorHydrogenExcitationBase::CalculateDifferentialCrossSection( const double anEnergy, const double cosTheta, double& aCrossSection )
    {
        // Nishimura et al., J. Phys. Soc. Jpn. 54 (1985) 1757.

        double a02 = KConst::BohrRadiusSquared();
        double clight = 1. / KConst::Alpha(); // velocity of light in atomic units is 1/ alpha

        double Cel[ 50 ] =
        { -0.512, -0.512, -0.509, -0.505, -0.499, -0.491, -0.476, -0.473, -0.462, -0.452, -0.438, -0.422, -0.406, -0.388, -0.370, -0.352, -0.333, -0.314, -0.296, -0.277, -0.258, -0.239, -0.221, -0.202, -0.185, -0.167, -0.151, -0.135, -0.120, -0.105, -0.092, -0.070, -0.053, -0.039, -0.030, -0.024, -0.019, -0.016, -0.014, -0.013, -0.012, -0.009, -0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.002, -0.001 };

        double e[ 10 ] =
        { 0., 3., 6., 12., 20., 32., 55., 85., 150., 250. };

        double t[ 10 ] =
        { 0., 10., 20., 30., 40., 60., 80., 100., 140., 180. };

        double D[ 9 ][ 10 ] =
        {
        { 2.9, 2.70, 2.5, 2.10, 1.80, 1.2000, 0.900, 1.0000, 1.600, 1.9 },
        { 4.2, 3.60, 3.1, 2.50, 1.90, 1.1000, 0.800, 0.9000, 1.300, 1.4 },
        { 6.0, 4.40, 3.2, 2.30, 1.80, 1.1000, 0.700, 0.5400, 0.500, 0.6 },
        { 6.0, 4.10, 2.8, 1.90, 1.30, 0.6000, 0.300, 0.1700, 0.160, 0.23 },
        { 4.9, 3.20, 2.0, 1.20, 0.80, 0.3000, 0.150, 0.0900, 0.050, 0.05 },
        { 5.2, 2.50, 1.2, 0.64, 0.36, 0.1300, 0.050, 0.0300, 0.016, 0.02 },
        { 4.0, 1.70, 0.7, 0.30, 0.16, 0.0500, 0.020, 0.0130, 0.010, 0.01 },
        { 2.8, 1.10, 0.4, 0.15, 0.07, 0.0200, 0.010, 0.0070, 0.004, 0.003 },
        { 1.2, 0.53, 0.2, 0.08, 0.03, 0.0074, 0.003, 0.0016, 0.001, 0.0008 } };

        double T, K2, K, d, st1, st2, DH, gam, CelK, Ki, theta;
        aCrossSection = -1.0;
        int i, j;
        T = anEnergy / 27.2;
        if( anEnergy >= 250. )
        {
            gam = 1. + T / (clight * clight); // relativistic correction factor
            K2 = 2. * T * (1. + gam) * (1. - cosTheta);
            if( K2 < 0. )
                K2 = 1.e-30;
            K = sqrt( K2 );
            if( K < 1.e-9 )
                K = 1.e-9; // momentum transfer
            d = 1.4009; // distance of protons in H2
            st1 = 8. + K2;
            st2 = 4. + K2;
            // DH is the diff. cross section for elastic electron scatt.
            // on atomic hydrogen within the first Born approximation :
            DH = 4. * st1 * st1 / (st2 * st2 * st2 * st2) * a02;
            // CelK calculation with linear interpolation.
            // CelK is the correction of the elastic electron
            // scatt. on molecular hydrogen compared to the independent atom
            // model.
            if( K < 3. )
            {
                i = (int) (K / 0.1); //WOLF int->double->int
                Ki = i * 0.1;
                CelK = Cel[ i ] + (K - Ki) / 0.1 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else if( K >= 3. && K < 5. )
            {
                i = (int) (30 + (K - 3.) / 0.2); //WOLF: int->double
                Ki = 3. + (i - 30) * 0.2; //WOLF: int->double
                CelK = Cel[ i ] + (K - Ki) / 0.2 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else if( K >= 5. && K < 9.49 )
            {
                i = (int) (40 + (K - 5.) / 0.5); //WOLF: int->double
                Ki = 5. + (i - 40) * 0.5; //WOLF: int->double
                CelK = Cel[ i ] + (K - Ki) / 0.5 * (Cel[ i + 1 ] - Cel[ i ]);
            }
            else
                CelK = 0.;

            aCrossSection = 2. * gam * gam * DH * (1. + sin( K * d ) / (K * d)) * (1. + CelK);
        } //end if anE>=250
        else
        {
            theta = acos( cosTheta ) * 180. / KConst::Pi();
            for( i = 0; i <= 8; i++ )
                if( anEnergy >= e[ i ] && anEnergy < e[ i + 1 ] )
                    for( j = 0; j <= 8; j++ )
                        if( theta >= t[ j ] && theta < t[ j + 1 ] )
                            aCrossSection = 1.e-20 * (D[ i ][ j ] + (D[ i ][ j + 1 ] - D[ i ][ j ]) * (theta - t[ j ]) / (t[ j + 1 ] - t[ j ]));
        }

        return;
    }

    /////////////////////////////////
    /////		Excitation BC	/////
    /////////////////////////////////

    KSIntCalculatorHydrogenExcitationBC::KSIntCalculatorHydrogenExcitationBC()
    {
    }

    KSIntCalculatorHydrogenExcitationBC::KSIntCalculatorHydrogenExcitationBC( const KSIntCalculatorHydrogenExcitationBC& )
    {
    }

    KSIntCalculatorHydrogenExcitationBC* KSIntCalculatorHydrogenExcitationBC::Clone() const
    {
        return new KSIntCalculatorHydrogenExcitationBC( *this );
    }

    KSIntCalculatorHydrogenExcitationBC::~KSIntCalculatorHydrogenExcitationBC()
    {
    }

    void KSIntCalculatorHydrogenExcitationBC::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {

        double aB[ 9 ] =
        { -4.2935194e2, 5.1122109e2, -2.8481279e2, 8.8310338e1, -1.6659591e1, 1.9579609, -1.4012824e-1, 5.5911348e-3, -9.5370103e-5 };
        double aC[ 9 ] =
        { -8.1942684e2, 9.8705099e2, -5.3095543e2, 1.5917023e2, -2.9121036e1, 3.3321027, -2.3305961e-1, 9.1191781e-3, -1.5298950e-4 };
        double lnsigma, lnE, lnEn, sigmaB, Emin, sigma, sigmaC;
        int n;
        sigma = 0.;
        Emin = 12.5;
        lnE = log( anEnergie );
        lnEn = 1.;
        lnsigma = 0.;
        if( anEnergie < Emin )
            sigmaB = 0.;
        else
        {
            for( n = 0; n <= 8; n++ )
            {
                lnsigma += aB[ n ] * lnEn;
                lnEn = lnEn * lnE;
            }
            sigmaB = exp( lnsigma );
        }
        sigma += sigmaB;
        //  sigma=0.;
        // C state:
        Emin = 15.8;
        lnE = log( anEnergie );
        lnEn = 1.;
        lnsigma = 0.;
        if( anEnergie < Emin )
            sigmaC = 0.;
        else
        {
            for( n = 0; n <= 8; n++ )
            {
                lnsigma += aC[ n ] * lnEn;
                lnEn = lnEn * lnE;
            }
            sigmaC = exp( lnsigma );
        }
        sigma += sigmaC;
        aCrossSection = sigma * 1.e-4;
    }

    void KSIntCalculatorHydrogenExcitationBC::CalculateEloss( const double, const double, double& anEloss )
    {
        anEloss = -0.045;
    }

    /////////////////////////////////
    /////		Ionisation		/////
    /////////////////////////////////

    KSIntCalculatorHydrogenIonisationOld::KSIntCalculatorHydrogenIonisationOld()
    {
    }

    KSIntCalculatorHydrogenIonisationOld::KSIntCalculatorHydrogenIonisationOld( const KSIntCalculatorHydrogenIonisationOld& )
    {
    }

    KSIntCalculatorHydrogenIonisationOld* KSIntCalculatorHydrogenIonisationOld::Clone() const
    {
        return new KSIntCalculatorHydrogenIonisationOld( *this );
    }

    KSIntCalculatorHydrogenIonisationOld::~KSIntCalculatorHydrogenIonisationOld()
    {
    }

    void KSIntCalculatorHydrogenIonisationOld::CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection )
    {
        CalculateCrossSection( aParticle.GetKineticEnergy_eV(), aCrossSection );
        return;
    }

    void KSIntCalculatorHydrogenIonisationOld::CalculateCrossSection( const double anEnergie, double& aCrossSection )
    {
//        This function computes the total ionization cross section of
//        electron scatt. on molecular hydrogen of
//        e+H2 --> e+e+H2^+  or  e+e+H^+ +H
//        anE<250 eV: Eq. 5 of J. Chem. Phys. 104 (1996) 2956
//        anE>250: sigma_i formula on page 107 in
//        Phys. Rev. A7 (1973) 103.
//        Good agreement with measured results of
//        PR A 54 (1996) 2146, and
//        Physica 31 (1965) 94.

        const double a02 = KConst::BohrRadiusSquared();
        const double ERyd = KConst::ERyd_eV();

        const double tBindingEnergy = 15.43;
        const double tOrbitalEnergy = 15.98;
        const int tNOccupation = 2;
        const double tDifferentFormula = 250.0;

        if( anEnergie > tOrbitalEnergy )
        {
            if( anEnergie > tDifferentFormula )
            {
                aCrossSection = 4. * KConst::Pi() * a02 * ERyd / anEnergie * (0.82 * log( anEnergie / ERyd ) + 1.3);
            }
            else
            {
                double t = anEnergie / tBindingEnergy;
                double u = tOrbitalEnergy / tBindingEnergy;
                double r = ERyd / tBindingEnergy;
                double S = 4. * KConst::Pi() * a02 * tNOccupation * r * r;
                double lnt = log( t );

                aCrossSection = S / (t + u + 1.) * (lnt / 2. * (1. - 1. / (t * t)) + 1. - 1. / t - lnt / (t + 1.));
            }
        }
        else
        {
            aCrossSection = 1e-40;
        }
        return;
    }

    void KSIntCalculatorHydrogenIonisationOld::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries )
    {
        double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
        KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

        // outgoing primary

        const double tIonisationEnergy = 15.43;

        double tLostKineticEnergy;
        double tTheta;
        double tPhi;

        //todo:://here now some fancy formulas from ferencs EH2scat

        tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );
        tTheta = 0.; // just to keep my darling clang quiet and charming
        tLostKineticEnergy = 0.; // just to keep my darling clang quiet and charming

        KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
        KThreeVector tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        KThreeVector tFinalDirection = tInitialDirection.Magnitude() * (sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit()) + cos( tTheta ) * tInitialDirection.Unit());

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tFinalDirection );
        aFinalParticle.SetKineticEnergy_eV( tInitialKineticEnergy - tLostKineticEnergy );
        aFinalParticle.AddLabel( GetName() );

        // outgoing secondary

        tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) );
        tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );

        tOrthogonalOne = tInitialDirection.Orthogonal();
        tOrthogonalTwo = tInitialDirection.Cross( tOrthogonalOne );
        tFinalDirection = tInitialDirection.Magnitude() * (sin( tTheta ) * (cos( tPhi ) * tOrthogonalOne.Unit() + sin( tPhi ) * tOrthogonalTwo.Unit()) + cos( tTheta ) * tInitialDirection.Unit());

        KSParticle* tSecondary = KSParticleFactory::GetInstance().Create( 11 );
        (*tSecondary) = anInitialParticle;
        tSecondary->SetMomentum( tFinalDirection );
        tSecondary->SetKineticEnergy_eV( tLostKineticEnergy - tIonisationEnergy );
        tSecondary->AddLabel( GetName() );

        aSecondaries.push_back( tSecondary );

        return;

    }

    void KSIntCalculatorHydrogenIonisationOld::InitializeComponent()
    {
//    	KTextFile* tInputFile = katrin::CreateDataTextFile( "HydrogenIonisationOld.dat" );
//
//        if( tInputFile->Open( katrin::KFile::eRead ) == true )
//        {
//            fstream& inputfile = *(tInputFile->File());
//
//            while( !inputfile.eof() )
//            {
//
//                //does this remove comments? should work.
//                Char_t c = inputfile.peek();
//                if( c >= '0' && c < '9' )
//                {
//                    inputfile >> fBindingEnergy >> fOrbitalEnergy >> fNOccupation;
//                }
//                else
//                {
//                    Char_t dump[ 200 ];
//                    inputfile.getline( dump, 200 );
//                    intmsg_debug( "KSScatteringHydrogenIonisationOld::InitializeComponent "<<ret );
//                    intmsg_debug( "dumping " << dump << " because " << c <<" is not a number" << eom );
//                    continue;
//                }
//            }
//        }
//        else
//        {
//            intmsg( eError ) << "KSScatteringHydrogenIonisationOld::InitializeComponent "<<ret;
//            intmsg( eError ) << " Cant open inputfile < "<<tInputFile->GetName()<<" > " << eom;
//        }
//        tInputFile->Close();
//        return;
    }

} /* namespace Kassiopeia */

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntCalculatorHydrogenBuilder::~KComplexElement()
    {
    }

    static int sKSIntCalculatorHydrogenStructure = KSIntCalculatorHydrogenBuilder::Attribute< string >( "name" ) + KSIntCalculatorHydrogenBuilder::Attribute< bool >( "elastic" ) + KSIntCalculatorHydrogenBuilder::Attribute< bool >( "excitation" ) + KSIntCalculatorHydrogenBuilder::Attribute< bool >( "ionisation" );

    //static int sToolboxKSIntCalculatorHydrogen = KToolboxBuilder::ComplexElement< KSIntCalculatorHydrogenData >( "ksint_calculator_hydrogen" );

}
