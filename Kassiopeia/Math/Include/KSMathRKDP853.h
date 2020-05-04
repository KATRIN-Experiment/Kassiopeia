#ifndef Kassiopeia_KSMathRKDP853_h_
#define Kassiopeia_KSMathRKDP853_h_


#include "KSMathMessage.h"


/* The basis for this ODE solver is given in:
* "Solving Ordinary Differential Equations I: Non-stiff Problems"
*  Hairer, Norsett, Wanner
*  Second Revised Edition.
*  page 181--196
*  It is based off Hairer's implementation of the Dormand & Prince RK86 with a continuous extension
*  and includes the coefficients of the Dormand-Prince Runge-Kutta algorithm DOP853.
*  These coefficients have been obtained from the file dop853.c given on Hairer's website here:
*  http://www.unige.ch/~hairer/prog/nonstiff/cprog.tar
*  Downloaded: 8/12/15
*  We retain the copyright and information notices below:
*/

/******************************************************************************/

/*
Copyright (c) 2004, Ernst Hairer

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS
IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/******************************************************************************/

/*

This code computes the numerical solution of a system of first order ordinary
differential equations y'=f(x,y). It uses an explicit Runge-Kutta method of
order 8(5,3) due to Dormand & Prince with step size control and dense output.

Authors : E. Hairer & G. Wanner
	  Universite de Geneve, dept. de Mathematiques
	  CH-1211 GENEVE 4, SWITZERLAND
	  E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH

The code is described in : E. Hairer, S.P. Norsett and G. Wanner, Solving
ordinary differential equations I, nonstiff problems, 2nd edition,
Springer Series in Computational Mathematics, Springer-Verlag (1993).

Version of Mai 2, 1994.

Remarks about the C version : this version allocates memory by itself, the
iwork array (among the initial FORTRAN parameters) has been splitted into
independant initial parameters, the statistical variables and last step size
and x have been encapsulated in the module and are now accessible through
dedicated functions; the variable names have been kept to maintain a kind
of reading compatibility between the C and FORTRAN codes; adaptation made by
J.Colinge (COLINGE@DIVSUN.UNIGE.CH).
*/

/******************************************************************************/

#include "KSMathIntegrator.h"

#include <cmath>
#include <limits>


#define KSMATHRKDP853_STAGE          12
#define KSMATHRKDP853_INTERP_STAGE   16
#define KSMATHRKDP853_EXTENDED_STAGE 3
#define KSMATHRKDP853_INTERP_ORDER   8

namespace Kassiopeia
{

template<class XSystemType> class KSMathRKDP853 : public KSMathIntegrator<XSystemType>
{
  public:
    KSMathRKDP853();
    ~KSMathRKDP853() override;

  public:
    typedef XSystemType SystemType;
    typedef KSMathDifferentiator<SystemType> DifferentiatorType;
    typedef typename SystemType::ValueType ValueType;
    typedef typename SystemType::DerivativeType DerivativeType;
    typedef typename SystemType::ErrorType ErrorType;

  public:
    void Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep,
                   ValueType& aFinalValue, ErrorType& anError) const override;


    /*******************************************************************/
    void ClearState() override
    {
        fHaveCachedDerivative = false;
    };

    //returns true if information valid
    bool GetInitialDerivative(DerivativeType& derv) const override
    {
        if (fHaveCachedDerivative) {
            derv = fDerivatives[0];
            return true;
        }
        return false;
    };

    //returns true if information valid
    bool GetFinalDerivative(DerivativeType& derv) const override
    {
        if (fHaveCachedDerivative) {
            derv = fDerivatives[KSMATHRKDP853_STAGE];
            return true;
        }
        return false;
    };

    //these functions are provided if the integrator implements
    //a method to interpolate the solution between initial and final step values
    //only valid for interpolating values on the last integration step
    bool HasDenseOutput() const override
    {
        return true;
    };
    void Interpolate(double aStepFraction, ValueType& aValue) const override;

    /******************************************************************/

  private:
    mutable bool fHaveCachedDerivative;
    mutable double fIntermediateTime[KSMATHRKDP853_INTERP_STAGE];
    mutable ValueType fValues[KSMATHRKDP853_INTERP_STAGE];
    mutable ValueType fDenseTerms[KSMATHRKDP853_INTERP_ORDER];
    mutable DerivativeType fDerivatives[KSMATHRKDP853_INTERP_STAGE];

    mutable ValueType yerr;
    mutable ValueType y8;

    //parameters defining the Butcher Tableau and dense output stages and interpolant
    static const double fA[KSMATHRKDP853_STAGE][KSMATHRKDP853_STAGE];
    static const double fAExtended[KSMATHRKDP853_EXTENDED_STAGE][KSMATHRKDP853_INTERP_STAGE];
    static const unsigned int fAColumnLimit[KSMATHRKDP853_INTERP_STAGE];

    static const double fB8[KSMATHRKDP853_STAGE];
    static const double fBhh[3];
    static const double fBErr[KSMATHRKDP853_STAGE];
    static const double fC[KSMATHRKDP853_INTERP_STAGE];
    static const double fD4[KSMATHRKDP853_INTERP_STAGE];
    static const double fD5[KSMATHRKDP853_INTERP_STAGE];
    static const double fD6[KSMATHRKDP853_INTERP_STAGE];
    static const double fD7[KSMATHRKDP853_INTERP_STAGE];
};

template<class XSystemType> KSMathRKDP853<XSystemType>::KSMathRKDP853()
{
    for (unsigned int i = 0; i < KSMATHRKDP853_STAGE; i++) {
        fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
        fValues[i] = std::numeric_limits<double>::quiet_NaN();
        fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
    }
    fHaveCachedDerivative = false;
}

template<class XSystemType> KSMathRKDP853<XSystemType>::~KSMathRKDP853() {}

template<class XSystemType>
void KSMathRKDP853<XSystemType>::Integrate(double aTime, const DifferentiatorType& aTerm,
                                           const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue,
                                           ErrorType& anError) const
{
    //do first stage (0) explicitly to deal with possibility of cached data
    //init value and time
    fValues[0] = anInitialValue;
    fIntermediateTime[0] = aTime;


    //init solution estimates
    ValueType y8 = fValues[0];

    // for(unsigned int i=0; i<ValueType::sDimension; i++)
    // {
    //     mathmsg_debug( "rkdp853 y[" << 0 <<"]["<<i<<"] = "<<fValues[0][i]<< eom );
    // }

    //we check if we have cached the derivative from the last step
    if (fHaveCachedDerivative) {
        fDerivatives[0] = fDerivatives[KSMATHRKDP853_STAGE];
    }
    else {
        aTerm.Differentiate(fIntermediateTime[0], fValues[0], fDerivatives[0]);
    }

    // for(unsigned int i=0; i<ValueType::sDimension; i++)
    // {
    //     mathmsg_debug( "rkdp853 derv[" << 0 <<"]["<<i<<"] = "<<fDerivatives[0][i]<< eom );
    // }

    //add contribution to 8th order estimate
    y8 = y8 + aStep * fB8[0] * fDerivatives[0];

    //compute the value of each stage and
    //evaluation of the derivative at each stage
    for (unsigned int i = 1; i < KSMATHRKDP853_STAGE; i++) {
        //compute the time of this stage
        fIntermediateTime[i] = fIntermediateTime[0] + aStep * fC[i];

        //now compute the stage value
        fValues[i] = fValues[0];
        for (unsigned int j = 0; j < fAColumnLimit[i]; j++) {
            fValues[i] = fValues[i] + (aStep * fA[i][j]) * fDerivatives[j];
        }

        // for(unsigned int j=0; j<ValueType::sDimension; j++)
        // {
        //     mathmsg_debug( "rkdp853 y[" << i <<"]["<<j<<"] = "<<fValues[i][j]<< eom );
        // }


        //now compute the derivative term for this stage
        aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);


        // for(unsigned int j=0; j<ValueType::sDimension; j++)
        // {
        //     mathmsg_debug( "rkdp853 derv[" << i <<"]["<<j<<"] = "<<fDerivatives[i][j]<< eom );
        // }

        //add contribution to 8th order estimate
        y8 = y8 + aStep * fB8[i] * fDerivatives[i];
    }

    //we use the 8th order estimate for the solution (extrapolation)
    aFinalValue = y8;

    // for(unsigned int j=0; j<ValueType::sDimension; j++)
    // {
    //     mathmsg_debug( "rkdp853 y_final["<<j<<"] = "<<y8[j]<< eom );
    // }

    //now estimate the truncation error on the step (for stepsize control)
    yerr = 0.0;
    for (unsigned int i = 0; i < KSMATHRKDP853_STAGE; i++) {
        yerr = yerr + fBErr[i] * fDerivatives[i];
    }
    anError = aStep * yerr;

    // for(unsigned int j=0; j<ValueType::sDimension; j++)
    // {
    //     mathmsg_debug( "rkdp853 y_err["<<j<<"] = "<<yerr[j]<< eom );
    // }

    //evaluate the derivative at final point and cache it for the next
    //step (this derivative is also needed for the dense output interpolation)
    fIntermediateTime[KSMATHRKDP853_STAGE] = aTime + aStep;
    fValues[KSMATHRKDP853_STAGE] = aFinalValue;
    aTerm.Differentiate(fIntermediateTime[KSMATHRKDP853_STAGE], aFinalValue, fDerivatives[KSMATHRKDP853_STAGE]);
    fHaveCachedDerivative = true;

    // for(unsigned int j=0; j<ValueType::sDimension; j++)
    // {
    //     mathmsg_debug( "rkdp853 y_final_derv["<<j<<"] = "<<fDerivatives[KSMATHRKDP853_STAGE][j]<< eom );
    // }

    //evaluate the extra derivatives needed for the 7th order dense output
    for (unsigned int i = 13; i < KSMATHRKDP853_INTERP_STAGE; i++) {
        //compute the time of this stage
        fIntermediateTime[i] = fIntermediateTime[0] + aStep * fC[i];

        //now compute the stage value
        fValues[i] = fValues[0];
        for (unsigned int j = 0; j < fAColumnLimit[i]; j++) {
            fValues[i] = fValues[i] + (aStep * fAExtended[i - 13][j]) * fDerivatives[j];
        }

        // for(unsigned int j=0; j<ValueType::sDimension; j++)
        // {
        //     mathmsg_debug( "rkdp853 y_ext[" << i <<"]["<<j<<"] = "<<fValues[i][j]<< eom );
        // }

        //now compute the derivative term for this stage
        aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);

        // for(unsigned int j=0; j<ValueType::sDimension; j++)
        // {
        //     mathmsg_debug( "rkdp853 derv_ext[" << i <<"]["<<j<<"] = "<<fDerivatives[i][j]<< eom );
        // }
    }

    //now we make the dense term evaluations for the continuous output
    fDenseTerms[0] = fValues[0];
    fDenseTerms[1] = fValues[KSMATHRKDP853_STAGE] - fValues[0];
    fDenseTerms[2] = aStep * fDerivatives[0] - fDenseTerms[1];
    fDenseTerms[3] = fDenseTerms[1] - aStep * fDerivatives[KSMATHRKDP853_STAGE] - fDenseTerms[2];
    fDenseTerms[4] = 0.0;
    fDenseTerms[5] = 0.0;
    fDenseTerms[6] = 0.0;
    fDenseTerms[7] = 0.0;
    for (unsigned int i = 0; i < KSMATHRKDP853_INTERP_STAGE; i++) {
        DerivativeType sderv = fDerivatives[i];
        sderv = aStep * sderv;
        fDenseTerms[4] = fDenseTerms[4] + fD4[i] * sderv;
        fDenseTerms[5] = fDenseTerms[5] + fD5[i] * sderv;
        fDenseTerms[6] = fDenseTerms[6] + fD6[i] * sderv;
        fDenseTerms[7] = fDenseTerms[7] + fD7[i] * sderv;
    }
    return;
}

template<class XSystemType> void KSMathRKDP853<XSystemType>::Interpolate(double aStepFraction, ValueType& aValue) const
{
    double s = aStepFraction;
    double s1 = 1.0 - s;
    aValue = 0.0;
    for (unsigned int i = KSMATHRKDP853_INTERP_ORDER - 1; i > 0; i--) {
        double x;
        if (i % 2 == 1) {
            x = s;
        }
        else {
            x = s1;
        }

        aValue = aValue + fDenseTerms[i];
        aValue = x * aValue;
    }
    aValue = aValue + fDenseTerms[0];
}

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fC[KSMATHRKDP853_INTERP_STAGE] = {
    0.0,                                   //c1
    0.526001519587677318785587544488E-01,  //c2
    0.789002279381515978178381316732E-01,  //c3
    0.118350341907227396726757197510E+00,  //c4
    0.281649658092772603273242802490E+00,  //c5
    0.333333333333333333333333333333E+00,  //c6
    0.25E+00,                              //c7
    0.307692307692307692307692307692E+00,  //c8
    0.651282051282051282051282051282E+00,  //c9
    0.6E+00,                               //c10
    0.857142857142857142857142857142E+00,  //c11
    1.0,                                   //c12
    1.0,                                   //c13
    0.1E+00,                               //c14
    0.2E+00,                               //c15
    0.777777777777777777777777777778E+00   //c16
};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fB8[KSMATHRKDP853_STAGE] = {
    5.42937341165687622380535766363E-2,   //b1
    0.0,                                  //b2
    0.0,                                  //b3
    0.0,                                  //b4
    0.0,                                  //b5
    4.45031289275240888144113950566E0,    //b6
    1.89151789931450038304281599044E0,    //b7
    -5.8012039600105847814672114227E0,    //b8
    3.1116436695781989440891606237E-1,    //b9
    -1.52160949662516078556178806805E-1,  //b10
    2.01365400804030348374776537501E-1,   //b11
    4.47106157277725905176885569043E-2    //b12
};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fBErr[KSMATHRKDP853_STAGE] = {
    0.1312004499419488073250102996E-01,   //b1
    0.0,                                  //b2
    0.0,                                  //b3
    0.0,                                  //b4
    0.0,                                  //b5
    -0.1225156446376204440720569753E+01,  //b6
    -0.4957589496572501915214079952E+00,  //b7
    0.1664377182454986536961530415E+01,   //b8
    -0.3503288487499736816886487290E+00,  //b9
    0.3341791187130174790297318841E+00,   //b10
    0.8192320648511571246570742613E-01,   //b11
    -0.2235530786388629525884427845E-01   //b12
};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fA[KSMATHRKDP853_STAGE][KSMATHRKDP853_STAGE] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {5.26001519587677318785587544488E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.97250569845378994544595329183E-2,
     5.91751709536136983633785987549E-2,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {2.95875854768068491816892993775E-2,
     0.0,
     8.87627564304205475450678981324E-2,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {2.41365134159266685502369798665E-1,
     0.0,
     -8.84549479328286085344864962717E-1,
     9.24834003261792003115737966543E-1,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {3.7037037037037037037037037037E-2,
     0.0,
     0.0,
     1.70828608729473871279604482173E-1,
     1.25467687566822425016691814123E-1,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {3.7109375E-2,
     0.0,
     0.0,
     1.70252211019544039314978060272E-1,
     6.02165389804559606850219397283E-2,
     -1.7578125E-2,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {3.70920001185047927108779319836E-2,
     0.0,
     0.0,
     1.70383925712239993810214054705E-1,
     1.07262030446373284651809199168E-1,
     -1.53194377486244017527936158236E-2,
     8.27378916381402288758473766002E-3,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},
    {6.24110958716075717114429577812E-1,
     0.0,
     0.0,
     -3.36089262944694129406857109825E0,
     -8.68219346841726006818189891453E-1,
     2.75920996994467083049415600797E1,
     2.01540675504778934086186788979E1,
     -4.34898841810699588477366255144E1,
     0.0,
     0.0,
     0.0,
     0.0},
    {4.77662536438264365890433908527E-1,
     0.0,
     0.0,
     -2.48811461997166764192642586468E0,
     -5.90290826836842996371446475743E-1,
     2.12300514481811942347288949897E1,
     1.52792336328824235832596922938E1,
     -3.32882109689848629194453265587E1,
     -2.03312017085086261358222928593E-2,
     0.0,
     0.0,
     0.0},
    {-9.3714243008598732571704021658E-1,
     0.0,
     0.0,
     5.18637242884406370830023853209E0,
     1.09143734899672957818500254654E0,
     -8.14978701074692612513997267357E0,
     -1.85200656599969598641566180701E1,
     2.27394870993505042818970056734E1,
     2.49360555267965238987089396762E0,
     -3.0467644718982195003823669022E0,
     0.0,
     0.0},
    {2.27331014751653820792359768449E0,
     0.0,
     0.0,
     -1.05344954667372501984066689879E1,
     -2.00087205822486249909675718444E0,
     -1.79589318631187989172765950534E1,
     2.79488845294199600508499808837E1,
     -2.85899827713502369474065508674E0,
     -8.87285693353062954433549289258E0,
     1.23605671757943030647266201528E1,
     6.43392746015763530355970484046E-1,
     0.0}};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fAExtended[KSMATHRKDP853_EXTENDED_STAGE][KSMATHRKDP853_INTERP_STAGE] = {
    /*a14*/ {5.61675022830479523392909219681E-2,
             0.0,
             0.0,
             0.0,
             0.0,
             0.0,
             2.53500210216624811088794765333E-1,
             -2.46239037470802489917441475441E-1,
             -1.24191423263816360469010140626E-1,
             1.5329179827876569731206322685E-1,
             8.20105229563468988491666602057E-3,
             7.56789766054569976138603589584E-3,
             -8.298E-3,
             0.0,
             0.0,
             0.0},
    /*a15*/
    {3.18346481635021405060768473261E-2,
     0.0,
     0.0,
     0.0,
     0.0,
     2.83009096723667755288322961402E-2,
     5.35419883074385676223797384372E-2,
     -5.49237485713909884646569340306E-2,
     0.0,
     0.0,
     -1.08347328697249322858509316994E-4,
     3.82571090835658412954920192323E-4,
     -3.40465008687404560802977114492E-4,
     1.41312443674632500278074618366E-1,
     0.0,
     0.0},
    /*a16*/
    {-4.28896301583791923408573538692E-1,
     0.0,
     0.0,
     0.0,
     0.0,
     -4.69762141536116384314449447206E0,
     7.68342119606259904184240953878E0,
     4.06898981839711007970213554331E0,
     3.56727187455281109270669543021E-1,
     0.0,
     0.0,
     0.0,
     -1.39902416515901462129418009734E-3,
     2.9475147891527723389556272149E0,
     -9.15095847217987001081870187138E0,
     0.0},
};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fD4[KSMATHRKDP853_INTERP_STAGE] = {-0.84289382761090128651353491142E+01,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.56671495351937776962531783590E+00,
                                                                            -0.30689499459498916912797304727E+01,
                                                                            0.23846676565120698287728149680E+01,
                                                                            0.21170345824450282767155149946E+01,
                                                                            -0.87139158377797299206789907490E+00,
                                                                            0.22404374302607882758541771650E+01,
                                                                            0.63157877876946881815570249290E+00,
                                                                            -0.88990336451333310820698117400E-01,
                                                                            0.18148505520854727256656404962E+02,
                                                                            -0.91946323924783554000451984436E+01,
                                                                            -0.44360363875948939664310572000E+01};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fD5[KSMATHRKDP853_INTERP_STAGE] = {0.10427508642579134603413151009E+02,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.24228349177525818288430175319E+03,
                                                                            0.16520045171727028198505394887E+03,
                                                                            -0.37454675472269020279518312152E+03,
                                                                            -0.22113666853125306036270938578E+02,
                                                                            0.77334326684722638389603898808E+01,
                                                                            -0.30674084731089398182061213626E+02,
                                                                            -0.93321305264302278729567221706E+01,
                                                                            0.15697238121770843886131091075E+02,
                                                                            -0.31139403219565177677282850411E+02,
                                                                            -0.93529243588444783865713862664E+01,
                                                                            0.35816841486394083752465898540E+02};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fD6[KSMATHRKDP853_INTERP_STAGE] = {0.19985053242002433820987653617E+02,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            -0.38703730874935176555105901742E+03,
                                                                            -0.18917813819516756882830838328E+03,
                                                                            0.52780815920542364900561016686E+03,
                                                                            -0.11573902539959630126141871134E+02,
                                                                            0.68812326946963000169666922661E+01,
                                                                            -0.10006050966910838403183860980E+01,
                                                                            0.77771377980534432092869265740E+00,
                                                                            -0.27782057523535084065932004339E+01,
                                                                            -0.60196695231264120758267380846E+02,
                                                                            0.84320405506677161018159903784E+02,
                                                                            0.11992291136182789328035130030E+02};

template<class XSystemType>
const double KSMathRKDP853<XSystemType>::fD7[KSMATHRKDP853_INTERP_STAGE] = {-0.25693933462703749003312586129E+02,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            0.0,
                                                                            -0.15418974869023643374053993627E+03,
                                                                            -0.23152937917604549567536039109E+03,
                                                                            0.35763911791061412378285349910E+03,
                                                                            0.93405324183624310003907691704E+02,
                                                                            -0.37458323136451633156875139351E+02,
                                                                            0.10409964950896230045147246184E+03,
                                                                            0.29840293426660503123344363579E+02,
                                                                            -0.43533456590011143754432175058E+02,
                                                                            0.96324553959188282948394950600E+02,
                                                                            -0.39177261675615439165231486172E+02,
                                                                            -0.14972683625798562581422125276E+03};

//list of the max column for each row in the fA matrix
//at which and beyond all entries are zero
template<class XSystemType>
const unsigned int KSMathRKDP853<XSystemType>::fAColumnLimit[KSMATHRKDP853_INTERP_STAGE] =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

}  // namespace Kassiopeia

#endif
