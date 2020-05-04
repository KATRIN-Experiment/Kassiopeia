#ifndef ElasticFerencCalculator_h
#define ElasticFerencCalculator_h

/** @file
 @brief contains ElasticFerencCalculator
 @details
 <b>Revision History:</b>
 \verbatim
 Date       Name        Brief description
 -----------------------------------------------------
 2009-06-29 S. Mertens  first version
 \endverbatim
 */

/*!
 \class Kassiopeia::ElasticFerencCalculator

 \author F. Glueck. Put into this class by S.Mertens

 \brief  class to calculate elastic electron - hydrogen Scattering
 @details
 <b>Detailed Description:</b>
 This class contains methods to calculate:
 - elastric cross section in m^2
 - energy loss (in eV) and scattering angle (in rad) for elastic scattering
 - differential cross section for elastic scattering in m^2/steradian
 */

#include <cmath>
#include <iostream>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////

namespace Kassiopeia
{

class ElasticFerencCalculator
{

  public:
    //!constructor.
    ElasticFerencCalculator();

    //!destructor.
    ~ElasticFerencCalculator();

    /*!
             \brief elastric cross section in m^2

             This function computes the total elastic cross section of
             electron scatt. on molecular hydrogen.
             See: Liu, Phys. Rev. A35 (1987) 591,
             Trajmar, Phys Reports 97 (1983) 221.
             \param anE incident electron energy in eV

             */

    double sigmaeltot(double anE);
    double sigmael(double anE);
    double sigmavib(double anE);
    double sigmarot02(double anE);
    double sigmarot13(double anE);
    double sigmarot20(double anE);

    /*!
             \brief energy loss and scattering angle for elastic scattering

             This subroutine generates  energy loss and polar scatt. angle according to
             electron elastic scattering in molecular hydrogen.

             \param anE incident electron energy in eV.
             \param Eloss  returns energy loss in eV
             \param Theta  returns change of polar angle in radian
             */
    void randomel(double anE, double& Eloss, double& Theta);

  protected:
    /*!
             \brief differential cross section for elastic scattering in m^2/steradian

             This subroutine computes the differential cross section
             Del= d sigma/d Omega  of  elastic electron scattering
             on molecular hydrogen in m^2/steradian.
             See: Nishimura et al., J. Phys. Soc. Jpn. 54 (1985) 1757.
             \param anE electron kinetic energy in eV
             \param cosTheta  cos(Theta), where Theta is the polar scatt. angle

             \return differential cross section for elastic scattering in m^2/steradian
             */
    double DiffXSecEl(double anE, double cosTheta);
};
}  // namespace Kassiopeia
#endif  //ElasticFerencCalculator_h
