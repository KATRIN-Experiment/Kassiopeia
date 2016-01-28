#ifndef Kassiopeia_KSMathRKDP54_h_
#define Kassiopeia_KSMathRKDP54_h_

#include <limits>

#include "KSMathIntegrator.h"

/* The basis for this ODE solver is given by Dormand & Prince:
* @article{dormand1980family,
*  title={A family of embedded Runge-Kutta formulae},
*  author={Dormand, John R and Prince, Peter J},
*  journal={Journal of computational and applied mathematics},
*  volume={6},
*  number={1},
*  pages={19--26},
*  year={1980},
*  publisher={Elsevier}
*
* }
* or see also:
* "Solving Ordinary Differential Equations I: Non-stiff Problems"
*  Hairer, Norsett, Wanner
*  Second Revised Edition.
*  page 178, table 5.2 (DOPRI5)
*/

#define KSMATHRKDP54_STAGE 7
#define KSMATHRKDP54_INTERP_ORDER 5

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRKDP54 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRKDP54();
            virtual ~KSMathRKDP54();

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
                    derv = fDerivatives[KSMATHRKDP54_STAGE]; return true;
                }
                return false;
            };

            //these functions are provided if the integrator implements
            //a method to interpolate the solution between initial and final step values
            //only valid for interpolating values on the last integration step
            virtual bool HasDenseOutput() const {return true;};
            virtual void Interpolate(double aStepFraction, ValueType& aValue) const;

            /******************************************************************/

        private:

            void ComputeDenseCoefficients(double sigma, double* coeff) const;

            mutable bool fHaveCachedDerivative;
            mutable double fIntermediateTime[ KSMATHRKDP54_STAGE + 1 ];
            mutable ValueType fValues[ KSMATHRKDP54_STAGE + 1 ];
            mutable DerivativeType fDerivatives[ KSMATHRKDP54_STAGE + 1 ];

            //parameters for calculateing dense output interpolant
            mutable double fBDense[KSMATHRKDP54_STAGE + 1];
            static const double fD[ KSMATHRKDP54_STAGE + 1 ][ KSMATHRKDP54_INTERP_ORDER ];

            //parameters defining the Butcher Tableau
            static const double fA[ KSMATHRKDP54_STAGE ][ KSMATHRKDP54_STAGE];
            static const unsigned int fAColumnLimit[ KSMATHRKDP54_STAGE];
            static const double fB4[ KSMATHRKDP54_STAGE ];
            static const double fB5[ KSMATHRKDP54_STAGE ];
            static const double fC[ KSMATHRKDP54_STAGE ];

    };

    template< class XSystemType >
    KSMathRKDP54< XSystemType >::KSMathRKDP54()
    {
        for(unsigned int i=0; i<KSMATHRKDP54_STAGE+1; i++)
        {
            fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
            fValues[i] = std::numeric_limits<double>::quiet_NaN();
            fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
        }
        fHaveCachedDerivative = false;
    }

    template< class XSystemType >
    KSMathRKDP54< XSystemType >::~KSMathRKDP54()
    {
    }

    template< class XSystemType >
    void KSMathRKDP54< XSystemType >::Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
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
            fDerivatives[0] = fDerivatives[KSMATHRKDP54_STAGE];
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
        for(unsigned int i=1; i<KSMATHRKDP54_STAGE; i++)
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

        //we use the 4th order estimate for the solution (better error estimation)
        aFinalValue = y4;

        //now estimate the truncation error on the step (for stepsize control)
        anError = y4 - y5;

        //evaluate the derivative at final point and cache it for the next
        //step (this derivative is needed for the dense output interpolation)
        fIntermediateTime[KSMATHRKDP54_STAGE] = aTime + aStep;
        fValues[KSMATHRKDP54_STAGE] = aFinalValue;
        aTerm.Differentiate( fIntermediateTime[KSMATHRKDP54_STAGE], aFinalValue, fDerivatives[KSMATHRKDP54_STAGE] );
        fHaveCachedDerivative = true;

        return;
    }

    template< class XSystemType >
    void KSMathRKDP54< XSystemType >::Interpolate(double aStepFraction, ValueType& aValue) const
    {
        double h = aStepFraction*(fIntermediateTime[KSMATHRKDP54_STAGE] - fIntermediateTime[0]);
        ComputeDenseCoefficients(aStepFraction, fBDense);
        aValue = fValues[0];

        for(unsigned int i=0; i<KSMATHRKDP54_STAGE + 1; i++)
        {
            aValue = aValue + h*fBDense[i]*fDerivatives[i];
        }
    }

    template< class XSystemType >
    void KSMathRKDP54< XSystemType >::ComputeDenseCoefficients(double sigma, double* coeff) const
    {
        //compute the polynomials using horner's method
        for(unsigned int i=0; i<KSMATHRKDP54_STAGE+1; i++)
        {
            coeff[i] = 0.0;
            for(unsigned int j=0; j<KSMATHRKDP54_INTERP_ORDER - 1; j++)
            {
                coeff[i] += fD[i][j];
                coeff[i] *= sigma;
            }
            coeff[i] += fD[i][KSMATHRKDP54_INTERP_ORDER-1];
        }
    }

    //coefficients for the time-steps
    template< class XSystemType >
    const double KSMathRKDP54< XSystemType >::fC[KSMATHRKDP54_STAGE] =
    {0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0};

    //coefficients for the 5th order estimate
    template< class XSystemType >
    const double KSMathRKDP54< XSystemType >::fB5[KSMATHRKDP54_STAGE] =
    {5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100, 1.0/40.0};

    //coefficients for the 4th order estimate
    template< class XSystemType >
    const double KSMathRKDP54< XSystemType >::fB4[KSMATHRKDP54_STAGE] =
    {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0};

    //6 by 6 matrix of coefficients for the linear combination of the derivatives
    template< class XSystemType >
    const double KSMathRKDP54< XSystemType >::fA[KSMATHRKDP54_STAGE][KSMATHRKDP54_STAGE] =
    {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0 , 0.0},
        {19372.0/6561, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0},
        {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0},
        {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0}
    };

    //list of the max column for each row in the fA matrix
    //at which and beyond all entries are zero
    template< class XSystemType >
    const unsigned int KSMathRKDP54< XSystemType >::fAColumnLimit[KSMATHRKDP54_STAGE] =
    {0, 1, 2, 3, 4, 5, 6};

    //7 by 4 matrix of coefficients for the interpolant polynomials
    template< class XSystemType >
    const double KSMathRKDP54< XSystemType >::fD[ KSMATHRKDP54_STAGE + 1 ][ KSMATHRKDP54_INTERP_ORDER ] =
    {
        { 157015080.0/11282082432.0, -13107642775.0/11282082432.0, 34969693132.0/11282082432.0, -32272833064.0/11282082432.0, 1.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0 },
        { -100*15701508.0/32700410799.0, 100*914128567.0/32700410799.0, -100*2074956840.0/32700410799.0, 100*1323431896.0/32700410799.0, 0.0},
        { 25.0*94209048.0/5641041216.0, -25.0*1518414297.0/5641041216.0, 25.0*2460397220.0/5641041216.0, -25.0*889289856.0/5641041216.0, 0.0 },
        { -2187.0*52338360.0/199316789632.0, 2187.0*451824525.0/199316789632.0, -2187.0*687873124.0/199316789632.0, 2187.0*259006536.0/199316789632.0, 0.0 },
        { 11.0*106151040.0/2467955532.0, -11.0*661884105.0/2467955532.0, 11.0*946554244.0/2467955532.0, -11.0*361440756.0/2467955532.0, 0.0 },
        { -8293050.0/29380423.0, 90730570.0/29380423.0, -127201567.0/29380423.0, 44764047.0/29380423.0, 0.0 }
    };

}

#endif
