#ifndef Kassiopeia_KSGenShakeOff_h_
#define Kassiopeia_KSGenShakeOff_h_

/**
 @file
 @brief contains KSGenShakeOff
 @details
 <b>Revision History:</b>
 \verbatim
 Date       Name          Brief description
 -----------------------------------------------
 10.02.2010   kaefer/wandkowsky      First version
 \endverbatim
 */
/*!
 @class Kassiopeia::KSGenShakeOff
 @author kaefer

 @brief handles shake off energy generation

 @details
 <b>Detailed Description:</b><br>




 */

#include "KTextFile.h"

namespace Kassiopeia
{

using std::vector;

class KSGenShakeOff
{

  public:
    KSGenShakeOff();
    ~KSGenShakeOff();

    void CreateSO(vector<int>& vacancy, std::vector<double>& energy);
    void SetForceCreation(bool asetting)
    {
        fForceCreation = asetting;
    }

  protected:
    double DiceEnergy(double bindingEnergy, int vacancy);
    std::string fFilename;
    vector<int> fShell;
    vector<double> fBindE;
    vector<double> fSoProb;
    vector<double> fSoProbNorm;
    bool fForceCreation;

    bool ReadData();
    katrin::KTextFile* fDataFile;
};

}  // namespace Kassiopeia
#endif  // KSGenSHAKEOFF_H_
