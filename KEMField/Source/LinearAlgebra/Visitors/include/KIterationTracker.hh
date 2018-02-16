#ifndef KITERATIONTRACKER_DEF
#define KITERATIONTRACKER_DEF

#include <limits.h>

#include "KIterativeSolver.hh"
#include "KSurfaceContainer.hh"

#include <ctime>
#include <fstream>

namespace KEMField
{
  template <typename ValueType>
  class KIterationTracker : public KIterativeSolver<ValueType>::Visitor
  {
  public:
    KIterationTracker()
      : fWriteInterval(1),
	fSaveName("iterationTracker.txt"),
	fStampCounter(0),
	fMaxIterationStamps(UINT_MAX)
    { KIterativeSolver<ValueType>::Visitor::Interval(1); }
    virtual ~KIterationTracker() {}

    void WriteInterval(unsigned int i) { fWriteInterval = i; }
    void MaxIterationStamps(unsigned int i) { fMaxIterationStamps = i; }
    void SaveName(std::string fileName) { fSaveName = fileName; }

    void Initialize(KIterativeSolver<ValueType>&);
    void Visit(KIterativeSolver<ValueType>&);
    void Finalize(KIterativeSolver<ValueType>&);

  private:
    void ResetInterval();

    struct KIterationStamp
    {
      KIterationStamp() : fIteration(0),fTime(0),fResidualNorm(0.) {}
      KIterationStamp(unsigned int iteration,
		      unsigned int time,
		      double residualNorm) : fIteration(iteration),
					     fTime(time),
					     fResidualNorm(residualNorm) {}
      unsigned int fIteration;
      unsigned int fTime;
      double fResidualNorm;
    };

    typedef std::vector<KIterationStamp> IterationStatusVector;

    unsigned int fWriteInterval;
    std::string fSaveName;
    unsigned int fStampCounter;
    unsigned int fMaxIterationStamps;

    IterationStatusVector fIterationStatus;
  };

  template <typename ValueType>
  void KIterationTracker<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
  {
    // clear the contents of the previous instance of the file
    std::ofstream file(fSaveName.c_str());
    file << static_cast<unsigned int>(time(NULL)) << "\t"
	 << solver.Tolerance() << "\n";
    file.close();
  }

  template <typename ValueType>
  void KIterationTracker<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
  {
    fIterationStatus.
      push_back(KIterationStamp(solver.Iteration(),
				static_cast<unsigned int>(time(NULL)),
				solver.ResidualNorm()));
    fStampCounter++;

    if (fIterationStatus.size()%fWriteInterval == 0)
    {
      std::ofstream file(fSaveName.c_str(),std::fstream::app);

      for (typename IterationStatusVector::iterator it = fIterationStatus.begin();
	   it!=fIterationStatus.end();++it)
	file << (*it).fIteration << "\t"
	     << (*it).fTime << "\t"
	     << (*it).fResidualNorm << "\n";

      file.close();

      fIterationStatus.clear();

      if (fStampCounter > fMaxIterationStamps)
	ResetInterval();
    }
  }

  template <typename ValueType>
  void KIterationTracker<ValueType>::ResetInterval()
  {
    this->fInterval *= 2;

    std::ifstream oldfile(fSaveName.c_str());
    unsigned int nStamps = fStampCounter;
    fStampCounter = 0;

    unsigned int startTime;
    double tolerance;

    oldfile >> startTime;
    oldfile >> tolerance;

    KIterationStamp stamp;

    for (unsigned int i=0;i<nStamps;i++)
    {
      oldfile >> stamp.fIteration;
      oldfile >> stamp.fTime;
      oldfile >> stamp.fResidualNorm;

      if (i%2==1)
      {
	fIterationStatus.push_back(stamp);
	fStampCounter++;
      }
    }

    oldfile.close();

    std::ofstream file(fSaveName.c_str());
    file << startTime << "\t"
	 << tolerance << "\n";

    for (typename IterationStatusVector::iterator it = fIterationStatus.begin();
	 it!=fIterationStatus.end();++it)
      file << (*it).fIteration << "\t"
	   << (*it).fTime << "\t"
	   << (*it).fResidualNorm << "\n";

    file.close();

    fIterationStatus.clear();
  }

  template <typename ValueType>
  void KIterationTracker<ValueType>::Finalize(KIterativeSolver<ValueType>& solver)
  {
    fIterationStatus.
      push_back(KIterationStamp(solver.Iteration(),
				static_cast<unsigned int>(time(NULL)),
				solver.ResidualNorm()));

    std::ofstream file(fSaveName.c_str(),std::fstream::app);

    for (typename IterationStatusVector::iterator it = fIterationStatus.begin();
	 it!=fIterationStatus.end();++it)
      file << (*it).fIteration << "\t"
	   << (*it).fTime << "\t"
	   << (*it).fResidualNorm << "\n";

    file.close();

    fIterationStatus.clear();
  }
}

#endif /* KITERATIONTRACKER_DEF */
