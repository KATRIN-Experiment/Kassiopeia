#ifndef KITERATIVESOLVER_DEF
#define KITERATIVESOLVER_DEF

#include <limits.h>

#include "KVector.hh"

#include "KEMCout.hh"

namespace KEMField
{
  template <typename ValueType>
  class KIterativeSolver
  {
  public:
    KIterativeSolver() : fIteration(0), fResidualNorm(1.), fTolerance(1.e-8) {}
    virtual ~KIterativeSolver();

    virtual void SetTolerance(double d) { fTolerance = d; }

    virtual unsigned int Dimension() const = 0;
    virtual unsigned int Iteration() const { return fIteration; }
    virtual double ResidualNorm() const { return fResidualNorm; }
    virtual double Tolerance() const { return fTolerance; }

    void SetIteration(unsigned int i) { fIteration = i; }
    unsigned int GetIteration() const {return fIteration;};

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
      Visitor() : fInterval(UINT_MAX), fTerminate(false) {};
      virtual ~Visitor() {};
      virtual void Initialize(KIterativeSolver&) = 0;
      virtual void Visit(KIterativeSolver&) = 0;
      virtual void Finalize(KIterativeSolver&) = 0;

      void Interval(unsigned int i) { fInterval = i; }
      unsigned int Interval() const { return fInterval; }

      bool Terminate() const {return fTerminate;};

    protected:
      unsigned int fInterval;
      bool fTerminate;
    };

    void AddVisitor(Visitor* visitor) { fVisitors.push_back(visitor); }

  protected:
    void InitializeVisitors();
    void AcceptVisitors();
    void FinalizeVisitors();
    bool Terminate();

    std::vector<Visitor*> fVisitors;
  };

  template <typename ValueType>
  KIterativeSolver<ValueType>::~KIterativeSolver()
  {
    for (typename std::vector<Visitor*>::iterator it=fVisitors.begin();it!=fVisitors.end();++it)
      delete *it;
  }

  template <typename ValueType>
  void KIterativeSolver<ValueType>::InitializeVisitors()
  {
    for (typename std::vector<Visitor*>::const_iterator it=fVisitors.begin();it!=fVisitors.end();++it)
      (*it)->Initialize(*this);
  }

  template <typename ValueType>
  void KIterativeSolver<ValueType>::FinalizeVisitors()
  {
    for (typename std::vector<Visitor*>::const_iterator it=fVisitors.begin();it!=fVisitors.end();++it)
      (*it)->Finalize(*this);
  }

  template <typename ValueType>
  bool KIterativeSolver<ValueType>::Terminate()
  {
    for (typename std::vector<Visitor*>::const_iterator it=fVisitors.begin();it!=fVisitors.end();++it)
    {
        if( (*it)->Terminate() ){return true;}
    }
    return false;
  }

  template <typename ValueType>
  void KIterativeSolver<ValueType>::AcceptVisitors()
  {
    for (typename std::vector<Visitor*>::const_iterator it=fVisitors.begin();it!=fVisitors.end();++it)
      if (Iteration()%(*it)->Interval()==0)
	(*it)->Visit(*this);
  }

  template <typename ValueType>
  class KIterationDisplay : public KIterativeSolver<ValueType>::Visitor
  {
  public:
    KIterationDisplay() { KIterativeSolver<ValueType>::Visitor::Interval(1); }
    virtual ~KIterationDisplay() {}
    void Initialize(KIterativeSolver<ValueType>& solver);
    void Visit(KIterativeSolver<ValueType>& solver);
    void Finalize(KIterativeSolver<ValueType>& solver);
  };

  template <typename ValueType>
  void KIterationDisplay<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
  {
    KEMField::cout<<"Beginning iterative solve with target residual norm "<<solver.Tolerance()<<KEMField::endl;
  }

  template <typename ValueType>
  void KIterationDisplay<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
  {
    KEMField::cout<<"Iteration, |Residual|: "<<solver.Iteration()<<" "<<solver.ResidualNorm()<<"  \r";
    KEMField::cout.flush();
  }

  template <typename ValueType>
  void KIterationDisplay<ValueType>::Finalize(KIterativeSolver<ValueType>& solver)
  {
    KEMField::cout<<"Convergence complete after "<<solver.Iteration()<<" iterations, with |Residual|: "<<solver.ResidualNorm()<<KEMField::endl;
  }

}

#endif /* KITERATIVESOLVER_DEF */
