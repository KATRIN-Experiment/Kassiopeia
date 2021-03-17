#ifndef KMATHBRACKETINGSOLVER_H_
#define KMATHBRACKETINGSOLVER_H_

#include "gsl/gsl_errno.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_roots.h"

namespace katrin
{

class KMathBracketingSolver
{
  public:
    enum
    {
        eBisection = 0,
        eFalsePositive = 1,
        eBrent = 2
    };

  public:
    KMathBracketingSolver();
    ~KMathBracketingSolver();

  public:
    // non-const member functions of type void( const double&, double& )
    template<class XTarget>
    void Solve(unsigned int aType, XTarget* anObject, void (XTarget::*aMember)(const double&, double&), double aValue,
               double aLowArgument, double aHighArgument, double& anArgument) const;

    // const member functions of type void( const double&, double& )
    template<class XTarget>
    void Solve(unsigned int aType, const XTarget* anObject, void (XTarget::*aMember)(const double&, double&) const,
               double aValue, double aLowArgument, double aHighArgument, double& anArgument) const;

    // non-const member functions of type double( const double& )
    template<class XTarget>
    void Solve(unsigned int aType, XTarget* anObject, double (XTarget::*aMember)(const double&), double aValue,
               double aLowArgument, double aHighArgument, double& anArgument) const;

    // const member functions of type double( const double& )
    template<class XTarget>
    void Solve(unsigned int aType, const XTarget* anObject, double (XTarget::*aMember)(const double&) const,
               double aValue, double aLowArgument, double aHighArgument, double& anArgument) const;

  private:
    class Function
    {
      public:
        Function() = default;
        virtual ~Function() = default;

        virtual double Evaluate(double anArgument) = 0;
    };

    template<class XTarget> class ReferringMemberFunction : public Function
    {
      public:
        ReferringMemberFunction(XTarget*& anObject, void (XTarget::*& aMember)(const double&, double&),
                                const double& aValue) :
            fObject(anObject),
            fMember(aMember),
            fValue(aValue)
        {}
        ~ReferringMemberFunction() override = default;

        double Evaluate(double anArgument) override
        {
            double tResult;
            (fObject->*fMember)(anArgument, tResult);
            tResult = tResult - fValue;
            return tResult;
        }

      private:
        XTarget* fObject;
        void (XTarget::*fMember)(const double&, double&);
        const double fValue;
    };

    template<class XTarget> class ReferringConstMemberFunction : public Function
    {
      public:
        ReferringConstMemberFunction(const XTarget*& anObject, void (XTarget::*& aMember)(const double&, double&) const,
                                     const double& aValue) :
            fObject(anObject),
            fMember(aMember),
            fValue(aValue)
        {}
        ~ReferringConstMemberFunction() override = default;

        double Evaluate(double anArgument) override
        {
            double tResult;
            (fObject->*fMember)(anArgument, tResult);
            tResult = tResult - fValue;
            return tResult;
        }

      private:
        const XTarget* fObject;
        void (XTarget::*fMember)(const double&, double&) const;
        const double fValue;
    };

    template<class XTarget> class ReturningMemberFunction : public Function
    {
      public:
        ReturningMemberFunction(XTarget*& anObject, double (XTarget::*& aMember)(const double&), const double& aValue) :
            fObject(anObject),
            fMember(aMember),
            fValue(aValue)
        {}
        ~ReturningMemberFunction() override = default;

        double Evaluate(double anArgument) override
        {
            double tResult;
            tResult = (fObject->*fMember)(anArgument);
            tResult = tResult - fValue;
            return tResult;
        }

      private:
        XTarget* fObject;
        double (XTarget::*fMember)(const double&);
        const double fValue;
    };

    template<class XTarget> class ReturningConstMemberFunction : public Function
    {
      public:
        ReturningConstMemberFunction(const XTarget*& anObject, double (XTarget::*& aMember)(const double&) const,
                                     const double& aValue) :
            fObject(anObject),
            fMember(aMember),
            fValue(aValue)
        {}
        ~ReturningConstMemberFunction() override = default;

        double Evaluate(double anArgument) override
        {
            double tResult;
            tResult = (fObject->*fMember)(anArgument);
            tResult = tResult - fValue;
            return tResult;
        }

      private:
        const XTarget* fObject;
        double (XTarget::*fMember)(const double&) const;
        const double fValue;
    };

    static double Evaluate(double anArgument, void* aFunction)
    {
        return (reinterpret_cast<Function*>(aFunction)->Evaluate(anArgument));
    }

  private:
    class Algorithms
    {
      public:
        Algorithms();
        ~Algorithms();

        gsl_root_fsolver* operator[](unsigned int aType);

      private:
        gsl_root_fsolver* fTypes[3];
    };

    static Algorithms sAlgorithms;
};

template<class XTarget>
inline void KMathBracketingSolver::Solve(unsigned int aType, const XTarget* anObject,
                                         void (XTarget::*aMember)(const double&, double&) const, double aValue,
                                         double aLowArgument, double aHighArgument, double& anArgument) const
{
    ReferringConstMemberFunction<XTarget> tMember(anObject, aMember, aValue);

    gsl_root_fsolver* tAlgorithm = sAlgorithms[aType];
    gsl_function tFunction = {&Evaluate, &tMember};

    int tStatus;
    int tCount = 0;
    gsl_root_fsolver_set(tAlgorithm, &tFunction, aLowArgument, aHighArgument);
    do {
        tCount++;
        tStatus = gsl_root_fsolver_iterate(tAlgorithm);
        anArgument = gsl_root_fsolver_root(tAlgorithm);
        aLowArgument = gsl_root_fsolver_x_lower(tAlgorithm);
        aHighArgument = gsl_root_fsolver_x_upper(tAlgorithm);
        tStatus = gsl_root_test_interval(aLowArgument, aHighArgument, 1.e-15, 0.);
    } while ((tStatus == GSL_CONTINUE) && (tCount < 100));
}

template<class XTarget>
inline void KMathBracketingSolver::Solve(unsigned int aType, XTarget* anObject,
                                         void (XTarget::*aMember)(const double&, double&), double aValue,
                                         double aLowArgument, double aHighArgument, double& anArgument) const
{
    ReferringMemberFunction<XTarget> tMember(anObject, aMember, aValue);

    gsl_root_fsolver* tAlgorithm = sAlgorithms[aType];
    gsl_function tFunction = {&Evaluate, &tMember};

    int tStatus;
    int tCount = 0;
    gsl_root_fsolver_set(tAlgorithm, &tFunction, aLowArgument, aHighArgument);
    do {
        tCount++;
        tStatus = gsl_root_fsolver_iterate(tAlgorithm);
        anArgument = gsl_root_fsolver_root(tAlgorithm);
        aLowArgument = gsl_root_fsolver_x_lower(tAlgorithm);
        aHighArgument = gsl_root_fsolver_x_upper(tAlgorithm);
        tStatus = gsl_root_test_interval(aLowArgument, aHighArgument, 1.e-15, 0.);
    } while ((tStatus == GSL_CONTINUE) && (tCount < 100));
}

template<class XTarget>
inline void KMathBracketingSolver::Solve(unsigned int aType, XTarget* anObject,
                                         double (XTarget::*aMember)(const double&), double aValue, double aLowArgument,
                                         double aHighArgument, double& anArgument) const
{
    ReturningMemberFunction<XTarget> tMember(anObject, aMember, aValue);

    gsl_root_fsolver* tAlgorithm = sAlgorithms[aType];
    gsl_function tFunction = {&Evaluate, &tMember};

    int tStatus;
    int tCount = 0;
    gsl_root_fsolver_set(tAlgorithm, &tFunction, aLowArgument, aHighArgument);
    do {
        tCount++;
        tStatus = gsl_root_fsolver_iterate(tAlgorithm);
        anArgument = gsl_root_fsolver_root(tAlgorithm);
        aLowArgument = gsl_root_fsolver_x_lower(tAlgorithm);
        aHighArgument = gsl_root_fsolver_x_upper(tAlgorithm);
        tStatus = gsl_root_test_interval(aLowArgument, aHighArgument, 1.e-15, 0.);
    } while ((tStatus == GSL_CONTINUE) && (tCount < 100));
}

template<class XTarget>
inline void KMathBracketingSolver::Solve(unsigned int aType, const XTarget* anObject,
                                         double (XTarget::*aMember)(const double&) const, double aValue,
                                         double aLowArgument, double aHighArgument, double& anArgument) const
{
    ReturningConstMemberFunction<XTarget> tMember(anObject, aMember, aValue);

    gsl_root_fsolver* tAlgorithm = sAlgorithms[aType];
    gsl_function tFunction = {&Evaluate, &tMember};

    int tStatus;
    int tCount = 0;
    gsl_root_fsolver_set(tAlgorithm, &tFunction, aLowArgument, aHighArgument);
    do {
        tCount++;
        tStatus = gsl_root_fsolver_iterate(tAlgorithm);
        anArgument = gsl_root_fsolver_root(tAlgorithm);
        aLowArgument = gsl_root_fsolver_x_lower(tAlgorithm);
        aHighArgument = gsl_root_fsolver_x_upper(tAlgorithm);
        tStatus = gsl_root_test_interval(aLowArgument, aHighArgument, 1.e-15, 0.);
    } while ((tStatus == GSL_CONTINUE) && (tCount < 100));
}

}  // namespace katrin

#endif
