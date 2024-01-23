/*
* KElectrostaticImportField.hh
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#ifndef KElectrostaticIMPORTFIELD_HH_
#define KElectrostaticIMPORTFIELD_HH_

#include "KElectrostaticField.hh"


namespace KEMField {

class KElectrostaticImportField: public KElectrostaticField
{
  public:
    KElectrostaticImportField();
    ~KElectrostaticImportField() override;

    void SetFile(const std::string& aFile);
    void SetSize(const std::string& aSize);

    void SetXRange(const KFieldVector& aXRange);
    void SetYRange(const KFieldVector& aYRange);
    void SetZRange(const KFieldVector& aZRange);


    KFieldVector GetField(const KPosition& P) const {
        return ElectricFieldCore(P);
    }

  protected:
    double PotentialCore(const KPosition& P) const override;
    KFieldVector ElectricFieldCore(const KPosition& P) const override;

  private:

    void SaveFieldSamples();

    std::string fFile;
    int fSize;
    KFieldVector fXRange;
    KFieldVector fYRange;
    KFieldVector fZRange;

    int nx;
    int ny;
    int nz;
    
    double xmin;
    double xmax;

    double ymin;
    double ymax;

    double zmin;
    double zmax;

    double* fEx;
    double* fEy;
    double* fEz;
    double* fPhi;



    double x1;
    double x2;
    double dx;

    double y1;
    double y2;
    double dy;

    double z1;
    double z2;
    double dz;
};

} /* namespace KEMFIELD */

#endif  //KElectrostaticIMPORTFIELD_HH

