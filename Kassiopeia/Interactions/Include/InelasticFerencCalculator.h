#ifndef InelasticFerencCalculator_h
#define InelasticFerencCalculator_h

/** @file
 @brief contains InelasticFerencCalculator
 @details
 <b>Revision History:</b>
 \verbatim
 Date       Name        Brief description
 -----------------------------------------------------
 2009-06-29 S. Mertens  first version
 \endverbatim
 */

/*!
 \class Kassiopeia::InelasticFerencCalculator

 \author F. Glueck. Put into this class by S.Mertens

 \brief  class to calculate inelastic electron  - hydrogen Scattering

 @details
 <b>Detailed Description:</b>
 This class contains methods to calculate:
 public:
 - electronic excitation cross section in m^2
 - energy loss (in eV) and scattering angle (in rad) for electronic excitation scattering
 - ionization cross section in  m^2
 - energy loss (in eV) and scattering angle (in rad) for ionization scattering

 protected:
 - differential cross section  of  inelastic electron scattering
 - total inelastic cross section
 - sigmaexc electronic excitation cross section to the B and C states
 - sigmadiss10 electronic dissociative excitation cross section
 - sigmadiss15 electronic dissociative excitation cross section
 - secondary electron energy


 The value for the inelastic cross section at 18.6 is 3.416E-22 m-2, in good agreement with the measured value of 3.40E-22 cm-2,
 (Assev et al) and calculated values (3.456E-22 m-2)
 */

#include <iostream>
#include <cmath>
#include <vector>

#include "Rtypes.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TMath.h"
#include "KTextFile.h"

//////////////////////////////////////////////////////////////////////////////////

namespace Kassiopeia
{
    class InelasticFerencCalculator
    {
        public:

            // constructor
            InelasticFerencCalculator();

            //!destructor.
            virtual ~InelasticFerencCalculator();

            /*!
             \brief set the molecule type for which calculations will be performed

             \param aMolecule std::string name of a registered molecule type
             */
            virtual void setmolecule( const std::string& aMolecule );

            /*!
             \brief returns the Ionization Energy of a secondary electron (a dice is thrown (weighted with the cross sections) to determine the shell from which the secondary electron is emmitted)
             Ionization Energy computed in function: sigmaion
             */
            virtual double GetIonizationEnergy();

            /*!
             \brief electronic excitation cross section in m^2

             This function computes the electronic excitation cross section of
             electron scatt. on molecular hydrogen in m^2 .
             \param anE incident electron energy in eV,
             \return  electronic excitation cross section in m^2
             */
            virtual double sigmaexc( double anE );

            /*!
             \brief energy loss and scattering angle for electronic excitation scattering

             This subroutine generates  energy loss and polar scatt. angle according to
             electron excitation scattering in molecular hydrogen.

             \param   anE incident electron energy in eV.
             \param   Eloss returns energy loss in eV
             \param   Theta returns change of polar angle in radian
             */
            virtual void randomexc( double anE, double& Eloss, double& Theta );

            /*!
             \brief ionization cross section in  m^2

             This function computes the total ionization cross section of
             electron scatt. on molecular hydrogen of
             e+H2 --> e+e+H2^+  or  e+e+H^+ +H
             anE<250 eV: Eq. 5 of J. Chem. Phys. 104 (1996) 2956
             anE>250: sigma_i formula on page 107 in
             Phys. Rev. A7 (1973) 103.
             Good agreement with measured results of
             PR A 54 (1996) 2146, and
             Physica 31 (1965) 94.

             \param anE incident electron energy in eV,
             \return  ionization cross section
             */
            virtual double sigmaion( double anE );

            /*!
             \brief energy loss and scattering angle for ionization scattering

             This subroutine generates  energy loss and polar scatt. angle according to
             electron ionization scattering in molecular hydrogen.
             The kinetic energy of the secondary electron is: Eloss-15.4 eV

             \param  anE incident electron energy in eV.
             \param  Eloss returns energy loss in eV
             \param  Theta returns change of polar angle in radian


             */
            virtual void randomion( double anE, double& Eloss, double& Theta );

        protected:
            inline double Lagrange( Int_t n, double *xn, double *fn, double x )
            {
                double f, a[100], b[100], aa, bb;
                f = 0.;
                for( Int_t j = 0; j < n; j++ )
                {
                    for( Int_t i = 0; i < n; i++ )
                    {
                        a[i] = x - xn[i];
                        b[i] = xn[j] - xn[i];
                    }
                    a[j] = b[j] = aa = bb = 1.;

                    for( Int_t i = 0; i < n; i++ )
                    {
                        aa = aa * a[i];
                        bb = bb * b[i];
                    }
                    f += fn[j] * aa / bb;
                }
                return f;
            }

            /*!
             \brief  total inelastic cross section


             This function computes the total inelastic cross section of  electron scatt. on molecular hydrogen,
             in the first Born approximation in m^2 .
             See: Liu, Phys. Rev. A35 (1987) 591.

             \param anE incident electron energy in eV,
             */
            double sigmainel( double anE );

            /*!
             \brief differential cross section of excitation electron scattering

             This subroutine computes the differential cross section
             Del= d sigma/d Omega  of excitation electron scattering
             on molecular hydrogen in m^2/steradian

             \param  anE electron kinetic energy in eV
             \param  cosTheta cos(Theta), where Theta is the polar scatt. angle

             \return  differential cross section of excitation electron scattering in m^2/steradian
             */
            double DiffXSecExc( double anE, double cosTheta );

            /*!
             \brief differential cross section  of  inelastic electron scattering


             This subroutine computes the differential cross section
             Dinel= d sigma/d Omega  of  inelastic electron scattering
             on molecular hydrogen,

             within the first Born approximation in m^2/steradian.

             \param anE electron kinetic energy in eV
             \param cosTheta cos(Theta), where Theta is the polar scatt. angle
             */
            double DiffXSecInel( double anE, double cosTheta );

            double sumexc( double K );

            /*!
             \brief sigmaexc electronic excitation cross section to the B and C states

             This function computes the sigmaexc electronic excitation
             cross section to the B and C states, with energy loss
             about 12.5 eV in m^2

             \param anE incident electron energy in eV,

             */
            double sigmaBC( double anE );

            /*!
             \brief sigmadiss10 electronic
             dissociative excitation

             This function computes the sigmadiss10 electronic
             dissociative excitation
             cross section, with energy loss
             about 10 eV in m^2
             \param anE  incident electron energy in eV,

             */
            double sigmadiss10( double anE );

            /*!
             \brief sigmadiss15 electronic dissociative excitation
             cross section

             This function computes the sigmadiss15 electronic dissociative excitation
             cross section, with energy loss about 15 eV  in m^2

             \param anE incident electron energy in eV
             */
            double sigmadiss15( double anE );

            /*!
             \brief secondary electron energy W

             This subroutine generates secondary electron energy W
             from ionization of incident electron energy E, by using
             the Lorentzian of Aseev  et al. (Eq. 8).
             E and W in eV.
             \param E  incident electron energy in eV
             \param W  returns secondary electron energy
             */
            void gensecelen( double E, double& W );

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            // the following functions and methods are intended for generalizing the cross sections.
            bool ReadData();
            bool FindMinimum(); //determines fminimum;

            katrin::KTextFile* fDataFile;

            std::vector< double > fBindingEnergy; // binding energy
            std::vector< double > fOrbitalEnergy; // orbital kinetic energy;
            std::vector< int > fNOccupation; //orbital occupation number

            double fIonizationEnergy;
            std::string fMoleculeType;
            int fMinimum; //position of orbital with minimal energy.
    };
}
#endif //InelasticFerencCalculator_h
