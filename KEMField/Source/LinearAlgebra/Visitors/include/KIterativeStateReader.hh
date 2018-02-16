#ifndef KITERATIVESTATEREADER_DEF
#define KITERATIVESTATEREADER_DEF

#include <deque>
#include <sstream>
#include <cstdio>
#include <limits>

#include "KIterativeSolver.hh"
#include "KSurfaceContainer.hh"

#include "KIterativeStateWriter.hh"

#include "KEMFileInterface.hh"

#include "KMD5HashGenerator.hh"

#include "KResidualVector.hh"

namespace KEMField
{
  template <typename ValueType>
  class KIterativeStateReader : public KIterativeSolver<ValueType>::Visitor
  {
  public:
    KIterativeStateReader(KSurfaceContainer& container)
    : fContainer(container) {}
    virtual ~KIterativeStateReader() {}

    void Initialize(KIterativeSolver<ValueType>&);
    void Visit(KIterativeSolver<ValueType>&) {}
    void Finalize(KIterativeSolver<ValueType>&) {}

  private:
    KSurfaceContainer& fContainer;
    KResidualVector<ValueType> fResidualVector;
    std::string fReadName;

    KMD5HashGenerator fHashGenerator;
  };

  template <typename ValueType>
  void KIterativeStateReader<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
  {
    std::vector< std::string > labels;

    KMD5HashGenerator hashGenerator;
    labels.push_back(hashGenerator.GenerateHash(fContainer));
    labels.push_back(KResidualVector<ValueType>::Name());

    if (KEMFileInterface::GetInstance()->NumberWithLabels(labels))
    {
      KEMFileInterface::GetInstance()->FindByLabels(fResidualVector,labels);
      solver.SetResidualVector(fResidualVector);
    }
  }
}

#endif /* KITERATIVESTATEREADER_DEF */
