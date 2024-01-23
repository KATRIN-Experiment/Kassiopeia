/*
* KMagnetostaticExpoField.hh
*
*  Created on: 08 Nov 2017
*      Author: alfredo
*      Modified: wonyongc
*/

//#ifndef KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELD_HH_
//#define KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELD_HH_

#ifndef KMAGNETOSTATICEXPOFIELD_HH_
#define KMAGNETOSTATICEXPOFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField {

class KMagnetostaticExpoField: public KMagnetostaticField
{

 public:
   KMagnetostaticExpoField() :
       KMagnetostaticField(),
       fB0(),
       fLambda() {}

   KMagnetostaticExpoField(const double& b0, const double& lambda) :
       KMagnetostaticField(),
       fB0(b0),
       fLambda(lambda) {}

   virtual ~KMagnetostaticExpoField() {}

 public:

   void SetB0(double aB0) { fB0 = aB0; }
   double GetB0() const { return fB0; }

   void SetLambda(double aLambda) { fLambda = aLambda; }
   double GetLambda() const { return fLambda; }

   KFieldVector GetField(const KPosition& P) const {
       return MagneticFieldCore(P);
   }

 private:
   KFieldVector MagneticPotentialCore(const KPosition& P) const {
       KFieldVector fval;
       fval[0]= fB0*cos(P[0]/fLambda)*exp(-P[2]/fLambda);
       fval[1]= 0.0;
       //fval[2]= -fval[0]*fLambda*abs(P[0]);
       fval[2]=-fB0*sin(P[0]/fLambda)*exp(-P[2]/fLambda);
       return 0.5 * fval.Cross(P);
   }

   KFieldVector MagneticFieldCore(const KPosition& P) const {
       KFieldVector fval;
       fval[0]= fB0*cos(P[0]/fLambda)*exp(-P[2]/fLambda);
       fval[1]= 0;
       //fval[2]= -fval[0]*fLambda*abs(P[0]);
       fval[2]=-fB0*sin(P[0]/fLambda)*exp(-P[2]/fLambda);
       return fval;
   }

   KGradient MagneticGradientCore(const KPosition& P) const {
       KGradient aGradient = KGradient(
           (-fB0/fLambda)*sin(P[0]/fLambda)*exp(-P[2]/fLambda),
           0.,
           (-fB0/fLambda)*cos(P[0]/fLambda)*exp(-P[2]/fLambda),
           0.,
           0.,
           0.,
           (-fB0/fLambda)*cos(P[0]/fLambda)*exp(-P[2]/fLambda),
           0.,
           (fB0/fLambda)*sin(P[0]/fLambda)*exp(-P[2]/fLambda)
       );
       return aGradient;
   }
 private:
   double fB0, fLambda;
//   KFieldVector fBx;
};

} /* namespace KEMFIELD */

#endif /* KMAGNETOSTATICEXPOFIELD_HH_ */

///* KEMFIELD_SOURCE_2_0_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELD_HH_ */
