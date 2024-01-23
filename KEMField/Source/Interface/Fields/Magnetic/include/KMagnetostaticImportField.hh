/*
* KMagnetostaticImportField.hh
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#ifndef KMagnetostaticIMPORTFIELD_HH_
#define KMagnetostaticIMPORTFIELD_HH_

#include "KMagnetostaticField.hh"


namespace KEMField {

class KMagnetostaticImportField: public KMagnetostaticField
{
  public:
    KMagnetostaticImportField();
    ~KMagnetostaticImportField() override;

    void SetFile(const std::string& aFile);
    void SetSize(const std::string& aSize);

    void SetXRange(const KFieldVector& aXRange);
    void SetYRange(const KFieldVector& aYRange);
    void SetZRange(const KFieldVector& aZRange);


    KFieldVector GetField(const KPosition& P) const {
        return MagneticFieldCore(P);
    }

  protected:
    KFieldVector MagneticPotentialCore(const KPosition& P) const override;
    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;

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

    double* fBx;
    double* fBy;
    double* fBz;



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

#endif  //KMagnetostaticIMPORTFIELD_HH

