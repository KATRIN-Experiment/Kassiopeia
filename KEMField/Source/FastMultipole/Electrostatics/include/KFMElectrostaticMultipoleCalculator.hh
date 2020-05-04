#ifndef KFMElectrostaticMultipoleCalculator_HH__
#define KFMElectrostaticMultipoleCalculator_HH__


#include "KFMMath.hh"
#include "KFMPoint.hh"
#include "KFMPointCloud.hh"
#include "KFMScalarMultipoleExpansion.hh"

#include <complex>
#include <vector>


namespace KEMField
{


/*
*
*@file KFMElectrostaticMultipoleCalculator.hh
*@class KFMElectrostaticMultipoleCalculator
*@brief abstract base class
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Dec 19 14:56:28 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleCalculator
{
  public:
    KFMElectrostaticMultipoleCalculator(){};
    virtual ~KFMElectrostaticMultipoleCalculator(){};

    virtual void SetDegree(int l_max) = 0;
    int GetDegree() const
    {
        return fDegree;
    };

    //constructs unscaled multipole expansion, assuming constant charge density
    //assumes a point cloud with 2 vertics is a wire electrode, 3 vertices is a triangle, and 4 is a rectangle/quadrilateral
    virtual bool ConstructExpansion(double* target_origin, const KFMPointCloud<3>* vertices,
                                    KFMScalarMultipoleExpansion* moments) const = 0;

  protected:
    int fDegree;
};

}  // namespace KEMField

#endif /* KFMElectrostaticMultipoleCalculator_H__ */
