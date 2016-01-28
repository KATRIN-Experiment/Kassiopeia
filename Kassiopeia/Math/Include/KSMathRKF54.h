#ifndef Kassiopeia_KSMathRKF54_h_
#define Kassiopeia_KSMathRKF54_h_

#include <limits>

#include "KSMathIntegrator.h"

/* The basis for this ODE solver is given in;
* (see table III)
*
* Erwin Fehlberg (1969).
* Low-order classical Runge-Kutta formulas with step size control
* and their application to some heat transfer problems.
* NASA Technical Report 315.
*
* or
*
* "Klassische Runge-Kutta-Formeln vierter und niedrigerer
* 0rdnung mit Schrittweiten-Kontrolle und ihre Anwendung
* auf Warmeleitungsprobleme"
* Journal:	Computing
* Publisher:	Springer Wien
* ISSN:	0010-485X (Print) 1436-5057 (Online)
* Issue:	Volume 6, Numbers 1-2 / March, 1970
* Pages	61-71
*/

#define KSMATHRKF54_STAGE 6

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRKF54 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRKF54();
            virtual ~KSMathRKF54();

        public:
            typedef XSystemType SystemType;
            typedef KSMathDifferentiator< SystemType > DifferentiatorType;
            typedef typename SystemType::ValueType ValueType;
            typedef typename SystemType::DerivativeType DerivativeType;
            typedef typename SystemType::ErrorType ErrorType;

        public:


        public:
            virtual void Integrate( double aTime,
                                    const DifferentiatorType& aTerm,
                                    const ValueType& anInitialValue,
                                    const double& aStep,
                                    ValueType& aFinalValue,
                                    ErrorType& anError ) const;


            /*******************************************************************/
            virtual void ClearState()
            {
                fHaveCachedDerivative = false;
            };

            //returns true if information valid
            virtual bool GetInitialDerivative(DerivativeType& derv) const
            {
                if(fHaveCachedDerivative)
                {
                    derv = fDerivatives[0]; return true;
                }
                return false;
            };

            //returns true if information valid
            virtual bool GetFinalDerivative(DerivativeType& derv) const
            {
                if(fHaveCachedDerivative)
                {
                    derv = fDerivatives[KSMATHRKF54_STAGE]; return true;
                }
                return false;
            };

            /******************************************************************/


        private:

            mutable bool fHaveCachedDerivative;
            mutable double fIntermediateTime[ KSMATHRKF54_STAGE +1 ];
            mutable ValueType fValues[ KSMATHRKF54_STAGE + 1 ];
            mutable DerivativeType fDerivatives[ KSMATHRKF54_STAGE +1 ];

            //parameters defining the Butcher Tableau
            static const double fA[ KSMATHRKF54_STAGE ][ KSMATHRKF54_STAGE];
            static const unsigned int fAColumnLimit[ KSMATHRKF54_STAGE];
            static const double fB4[ KSMATHRKF54_STAGE ];
            static const double fB5[ KSMATHRKF54_STAGE ];
            static const double fC[ KSMATHRKF54_STAGE ];

    };

    template< class XSystemType >
    KSMathRKF54< XSystemType >::KSMathRKF54()
    {
        for(unsigned int i=0; i<KSMATHRKF54_STAGE+1; i++)
        {
            fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
            fValues[i] = std::numeric_limits<double>::quiet_NaN();
            fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
        }
        fHaveCachedDerivative = false;
    }

    template< class XSystemType >
    KSMathRKF54< XSystemType >::~KSMathRKF54()
    {
    }

    template< class XSystemType >
    void KSMathRKF54< XSystemType >::Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
    {

        //do first stage (0) explicitly to deal with possibility of cached data
        //init value and time
        fValues[0] = anInitialValue;
        fIntermediateTime[0] = aTime;

        //init solution estimates
        ValueType y4 = fValues[0];
        ValueType y5 = fValues[0];

        //we check if we have cached the derivative from the last step
        if( fHaveCachedDerivative )
        {
            fDerivatives[0] = fDerivatives[KSMATHRKF54_STAGE];
        }
        else
        {
            aTerm.Differentiate( fIntermediateTime[0], fValues[0], fDerivatives[0] );
        }

        //add contribution to 4th order estimate
        y4 = y4 + aStep*fB4[0]*fDerivatives[0];
        //add contribution to 5th order estimate
        y5 = y5 + aStep*fB5[0]*fDerivatives[0];

        //compute the value of each stage and
        //evaluation of the derivative at each stage
        for(unsigned int i=1; i<KSMATHRKF54_STAGE; i++)
        {
            //compute the time of this stage
            fIntermediateTime[i] = fIntermediateTime[0] + aStep*fC[i];

            //now compute the stage value
            fValues[i] = fValues[0];
            for(unsigned int j=0; j<fAColumnLimit[i]; j++)
            {
                fValues[i] = fValues[i] + (aStep*fA[i][j])*fDerivatives[j];
            }

            //now compute the derivative term for this stage
            aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);

            //add contribution to 4th order estimate
            y4 = y4 + aStep*fB4[i]*fDerivatives[i];

            //add contribution to 5th order estimate
            y5 = y5 + aStep*fB5[i]*fDerivatives[i];
        }

        //we use the 5th order estimate for the solution (local extrapolation)
        aFinalValue = y5;

        //now estimate the truncation error on the step (for stepsize control)
        anError = y4 - y5;

        //evaluate the derivative at final point and cache it for the next
        //step (this derivative is needed for the dense output interpolation)
        fIntermediateTime[KSMATHRKF54_STAGE] = aTime + aStep;
        aTerm.Differentiate( fIntermediateTime[KSMATHRKF54_STAGE], aFinalValue, fDerivatives[KSMATHRKF54_STAGE] );
        fHaveCachedDerivative = true;

        return;
    }

    //coefficients for the time-steps
    template< class XSystemType >
    const double KSMathRKF54< XSystemType >::fC[KSMATHRKF54_STAGE] =
    {0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0};

    //coefficients for the 5th order estimate
    template< class XSystemType >
    const double KSMathRKF54< XSystemType >::fB5[KSMATHRKF54_STAGE] =
    {16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0};

    //coefficients for the 4th order estimate
    template< class XSystemType >
    const double KSMathRKF54< XSystemType >::fB4[KSMATHRKF54_STAGE] =
    {25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0};

    //6 by 6 matrix of coefficients for the linear combination of the derivatives
    template< class XSystemType >
    const double KSMathRKF54< XSystemType >::fA[KSMATHRKF54_STAGE][KSMATHRKF54_STAGE] =
    {
       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0},
       {1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0},
       {439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0},
       {-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0}
    };

    //list of the max column for each row in the fA matrix
    //at which and beyond all entries are zero
    template< class XSystemType >
    const unsigned int KSMathRKF54< XSystemType >::fAColumnLimit[KSMATHRKF54_STAGE] =
    {0, 1, 2, 3, 4, 5};

}

#endif
