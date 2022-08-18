#ifndef KITERATIVESOLVER_DEF
#define KITERATIVESOLVER_DEF

#include "KEMCoreMessage.hh"
#include "KVector.hh"

#include <climits>
#include <vector>

namespace KEMField
{
template<typename ValueType> class KIterativeSolver
{
  public:
    KIterativeSolver() : fIteration(0), fResidualNorm(1.), fTolerance(1.e-8) {}
    virtual ~KIterativeSolver();

    virtual void SetTolerance(double d)
    {
        fTolerance = d;
    }

    virtual unsigned int Dimension() const = 0;
    virtual unsigned int Iteration() const
    {
        return fIteration;
    }
    virtual double ResidualNorm() const
    {
        return fResidualNorm;
    }
    virtual double Tolerance() const
    {
        return fTolerance;
    }

    void SetIteration(unsigned int i)
    {
        fIteration = i;
    }
    unsigned int GetIteration() const
    {
        return fIteration;
    };

    virtual void SetResidualVector(const KVector<ValueType>&) {}
    virtual void GetResidualVector(KVector<ValueType>&) {}

    virtual void CoalesceData() {}

  protected:
    unsigned int fIteration;
    double fResidualNorm;
    double fTolerance;

  public:
    class Visitor
    {
      public:
        Visitor() : fInterval(UINT_MAX), fTerminate(false){};
        virtual ~Visitor() = default;
        ;
        virtual void Initialize(KIterativeSolver&) = 0;
        virtual void Visit(KIterativeSolver&) = 0;
        virtual void Finalize(KIterativeSolver&) = 0;

        void Interval(unsigned int i)
        {
            fInterval = i;
        }
        unsigned int Interval() const
        {
            return fInterval;
        }

        bool Terminate() const
        {
            return fTerminate;
        };

      protected:
        unsigned int fInterval;
        bool fTerminate;
    };

    void AddVisitor(Visitor* visitor)
    {
        fVisitors.push_back(visitor);
    }

  protected:
    void InitializeVisitors();
    void AcceptVisitors();
    void FinalizeVisitors();
    bool Terminate();

    std::vector<Visitor*> fVisitors;
};

template<typename ValueType> KIterativeSolver<ValueType>::~KIterativeSolver()
{
    for (auto it = fVisitors.begin(); it != fVisitors.end(); ++it)
        delete *it;
}

template<typename ValueType> void KIterativeSolver<ValueType>::InitializeVisitors()
{
    for (typename std::vector<Visitor*>::const_iterator it = fVisitors.begin(); it != fVisitors.end(); ++it)
        (*it)->Initialize(*this);
}

template<typename ValueType> void KIterativeSolver<ValueType>::FinalizeVisitors()
{
    for (typename std::vector<Visitor*>::const_iterator it = fVisitors.begin(); it != fVisitors.end(); ++it)
        (*it)->Finalize(*this);
}

template<typename ValueType> bool KIterativeSolver<ValueType>::Terminate()
{
    for (typename std::vector<Visitor*>::const_iterator it = fVisitors.begin(); it != fVisitors.end(); ++it) {
        if ((*it)->Terminate()) {
            return true;
        }
    }
    return false;
}

template<typename ValueType> void KIterativeSolver<ValueType>::AcceptVisitors()
{
    for (typename std::vector<Visitor*>::const_iterator it = fVisitors.begin(); it != fVisitors.end(); ++it) {
        if (Iteration() % (*it)->Interval() == 0)
            (*it)->Visit(*this);
    }
}

template<typename ValueType> class KIterationDisplay : public KIterativeSolver<ValueType>::Visitor
{
  public:
    KIterationDisplay() : fPrefix(""), fCarriageReturn(false)
    {
        KIterativeSolver<ValueType>::Visitor::Interval(1);
    }
    KIterationDisplay(const std::string& prefix) : fPrefix(prefix), fCarriageReturn(true)
    {
        KIterativeSolver<ValueType>::Visitor::Interval(1);
    }
    ~KIterationDisplay() override = default;
    void Initialize(KIterativeSolver<ValueType>& solver) override;
    void Visit(KIterativeSolver<ValueType>& solver) override;
    void Finalize(KIterativeSolver<ValueType>& solver) override;

  private:
    std::string fPrefix;
    bool fCarriageReturn;
};

template<typename ValueType> void KIterationDisplay<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
{
    kem_cout(eNormal) << fPrefix << "Beginning iterative solve with target residual norm " << solver.Tolerance()
               << " and dimension " << solver.Dimension() << eom;
}

template<typename ValueType> void KIterationDisplay<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
{
    kem_cout(eNormal) << fPrefix << "Iteration, |Residual|: " << solver.Iteration() << ", " << solver.ResidualNorm()
               << (fCarriageReturn ? ret : eom);
}

template<typename ValueType> void KIterationDisplay<ValueType>::Finalize(KIterativeSolver<ValueType>& solver)
{
    if (!std::isfinite(solver.ResidualNorm()) || solver.ResidualNorm() == 0) {
        kem_cout(eError) << fPrefix << "Convergence incomplete after " << solver.Iteration()
                   << " iterations, with |Residual|: " << solver.ResidualNorm() << eom;
    }

    kem_cout(eNormal) << fPrefix << "Convergence complete after " << solver.Iteration()
               << " iterations, with |Residual|: " << solver.ResidualNorm() << eom;
}

}  // namespace KEMField

#endif /* KITERATIVESOLVER_DEF */
