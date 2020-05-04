#ifndef Kassiopeia_KSGenRelaxation_h_
#define Kassiopeia_KSGenRelaxation_h_

/**
 @file
 @brief contains KSGenRelaxation
 @details
 <b>Revision History:</b>
 \verbatim
 Date       Name          Brief description
 -----------------------------------------------
 10.02.2010   kaefer/wandkowsky      First version
 24.04.2011   mertens                cleaned up a bit
 \endverbatim
 */

/*!

 @class Kassiopeia::KSGenRelaxation
 @author kaefer/renschler/mertens/wandkowsky

 @brief Applicable to singly ioized atoms, ie. with a vacancy in the K to x(depends on input file) shell. The class relaxes the atom by creating auger electrons and fluorecene photons until there is no vacancy left. The code is based on inputfiles by penelope (see docu page 230.)

 @details
 <b>Detailed Description:</b><br>


 */

#include "KTextFile.h"

#include <map>

namespace Kassiopeia
{

class KSGenRelaxation
{

  public:
    KSGenRelaxation();
    ~KSGenRelaxation();

    void ClearAugerEnergies()
    {
        fAugerEnergies.clear();
    }

    void ClearVacancies()
    {
        fVacancies->clear();
    }

    bool Initialize(int isotope);

    std::vector<double> GetAugerEnergies() const;
    std::vector<double> GetFluorescenceEnergies() const;
    std::vector<unsigned int>* GetVacancies() const;

    void RelaxVacancy(unsigned int shell);
    void Relax();
    void Relax(unsigned int vacancy);

    void SetIsotope(int isotope)
    {
        fIsotope = isotope;
    }

  protected:
    typedef struct line_struct
    {
        unsigned int vacOne, vacTwo, vacThree;
        double probability, energy;
    } line;

    bool ReadData();
    katrin::KTextFile* fDataFile;

    std::map<unsigned int, double> fshellEnergies;
    std::vector<line> fTransProp;

    std::vector<double> fAugerEnergies;
    std::vector<double> fFluorescenceEnergies;
    std::vector<unsigned int>* fVacancies;

    int fIsotope;
};

inline std::vector<double> KSGenRelaxation::GetAugerEnergies() const
{
    return fAugerEnergies;
}

inline std::vector<double> KSGenRelaxation::GetFluorescenceEnergies() const
{
    return fFluorescenceEnergies;
}

inline std::vector<unsigned int>* KSGenRelaxation::GetVacancies() const
{
    return fVacancies;
}

}  // namespace Kassiopeia

#endif /* KSGenRelaxation_H_ */
