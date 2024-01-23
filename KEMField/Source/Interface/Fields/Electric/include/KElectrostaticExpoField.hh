#ifndef KELECTROSTATICEXPOFIELD_DEF
#define KELECTROSTATICEXPOFIELD_DEF
#include "KElectrostaticField.hh"
#include <math.h>

namespace KEMField {

class KElectrostaticExpoField : public KElectrostaticField
{
  public:
    KElectrostaticExpoField() :
        KElectrostaticField(),
        fTKE(),
        fB0(),
        fLambda() {}

    KElectrostaticExpoField(const double& tke, const double& b0, const double& lambda) :
        KElectrostaticField(),
        fTKE(tke),
        fB0(b0),
        fLambda(lambda) {}

    virtual ~KElectrostaticExpoField() {}


//    void SetEy(KFieldVector ey) { fEy = ey; }
//    void SetEpar(KFieldVector epar) { fEpar = epar; }
//    KFieldVector GetEy() {return fEy;}
//    KFieldVector GetEpar() {return fEpar;}

    void SetTKE(double aTKE) { fTKE = aTKE; }
    double GetTKE() const { return fTKE; }

    void SetB0(double aB0) { fB0 = aB0; }
    double GetB0() const { return fB0; }

    void SetLambda(double aLambda) { fLambda = aLambda; }
    double GetLambda() const { return fLambda; }

    void SetY0(double aY0) { fY0 = aY0; }
    double GetY0() const { return fY0; }

    static std::string Name() { return "ElectrostaticExpoFieldSolver"; }

  private:
    virtual double PotentialCore(const KPosition& P) const {

//        KFieldVector fval;
//        fval[0]= (fEpar[0]/(fEpar[1]+(P[0]-fEpar[2]))) - (fEpar[0]/(fEpar[1]-(P[0]-fEpar[2])));
//        fval[1]= fEy[0]*cos(P[1]/fEy[1])*exp(-P[2]/fEy[1]);
//        fval[1]= fEy[0]*cos(P[1]/fEy[1])*exp(-P[2]/fEy[1]) + (2*fEpar[0]/(fEpar[1]*fEpar[1]))*(P[1]-fEpar[2]);
//        fval[2]= -fEy[0]*sin(P[1]/fEy[1])*exp(-P[2]/fEy[1]);
//        return fval.Dot(P);

        return fTKE*fB0*exp(-P[2]/fLambda)*sin(P[1]/fLambda)-fTKE*fB0;
    }

    virtual KFieldVector ElectricFieldCore(const KPosition& P) const {
        KFieldVector fval;

//        fval[0]= 0;
//        fval[1]= ( fEy[0] / ( fEy[1]* sin(fEy[2]/fEy[1] )  )  )*cos(P[1]/fEy[1])*exp(-P[2]/fEy[1]);
//        fval[2]=-( fEy[0] / ( fEy[1]* sin(fEy[2]/fEy[1] )  )  )*sin(P[1]/fEy[1])*exp(-P[2]/fEy[1]);

        fval[0]= 0;
        fval[1]= (fTKE*fB0/fLambda)*cos(P[1]/fLambda)*exp(-P[2]/fLambda);
        fval[2]=-(fTKE*fB0/fLambda)*sin(P[1]/fLambda)*exp(-P[2]/fLambda);

//        fval[0]= 0;
//        fval[1]= ((fTKE*fB0)/(fLambda*sin(fY0/fLambda)))*cos(P[1]/fLambda)*exp(-P[2]/fLambda);
//        fval[2]=-((fTKE*fB0)/(fLambda*sin(fY0/fLambda)))*sin(P[1]/fLambda)*exp(-P[2]/fLambda);

//        fval[0]= (fEpar[0]/(fEpar[1]+(P[0]-fEpar[2]))) - (fEpar[0]/(fEpar[1]-(P[0]-fEpar[2])));
//        fval[1]= fEy[0]*cos(P[1]/fEy[1])*exp(-P[2]/fEy[1]) + (2*fEpar[0]/(fEpar[1]*fEpar[1]))*(P[1]-fEpar[2]);

        return fval;
    }



  protected:
    double fTKE, fB0, fLambda, fY0;
//    KFieldVector fEy;
//    KFieldVector fEpar;

};

}
#endif /* KELECTROSTATICEXPOFIELD_DEF */
