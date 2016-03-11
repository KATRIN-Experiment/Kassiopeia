#ifndef Kassiopeia_KSMathIntegrator_h_
#define Kassiopeia_KSMathIntegrator_h_

#include "KSMathSystem.h"
#include "KSMathDifferentiator.h"

namespace Kassiopeia
{
    template< class XType >
    class KSMathIntegrator;

    template< class XValueType, class XDerivativeType, class XErrorType >
    class KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >
    {
        public:
            KSMathIntegrator();
            virtual ~KSMathIntegrator();

        public:
            virtual void Integrate( double aTime,
                                    const KSMathDifferentiator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >& aTerm,
                                    const XValueType& anInitialValue,
                                    const double& aStep,
                                    XValueType& aFinalValue,
                                    XErrorType& anError ) const = 0;

            //the below functions may only be meaningful for explicit Runge-Kutta type integrators
            //they are primarily intended for interpolation/dense output of the solution between mesh points

            /*******************************************************************/
            virtual void ClearState() {};

            //returns true if information valid
            virtual bool GetInitialDerivative(XDerivativeType& /*derv*/) const {return false;};

            //returns true if information valid
            virtual bool GetFinalDerivative(XDerivativeType& /*derv*/) const {return false;};

            //these functions are provided if the integrator implements
            //a method to interpolate the solution between initial and final step values
            //only valid for interpolating values on the last integration step
            virtual bool HasDenseOutput() const {return false;};
            virtual void Interpolate(double /*aStepFraction*/, XValueType& /*aValue*/) const {;};

            /******************************************************************/

    };

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::KSMathIntegrator()
    {
    }

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::~KSMathIntegrator()
    {
    }

}

#endif
