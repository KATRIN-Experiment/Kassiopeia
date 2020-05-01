#ifndef KITERATIVESTATEREADER_DEF
#define KITERATIVESTATEREADER_DEF

#include "KEMFileInterface.hh"
#include "KIterativeSolver.hh"
#include "KIterativeStateWriter.hh"
#include "KMD5HashGenerator.hh"
#include "KResidualVector.hh"
#include "KSurfaceContainer.hh"

#include <cstdio>
#include <deque>
#include <limits>
#include <sstream>

namespace KEMField
{
template<typename ValueType> class KIterativeStateReader : public KIterativeSolver<ValueType>::Visitor
{
  public:
    KIterativeStateReader(KSurfaceContainer& container) : fContainer(container) {}
    ~KIterativeStateReader() override {}

    void Initialize(KIterativeSolver<ValueType>&) override;
    void Visit(KIterativeSolver<ValueType>&) override {}
    void Finalize(KIterativeSolver<ValueType>&) override {}

  private:
    KSurfaceContainer& fContainer;
    KResidualVector<ValueType> fResidualVector;
    std::string fReadName;

    KMD5HashGenerator fHashGenerator;
};

template<typename ValueType> void KIterativeStateReader<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
{
    std::vector<std::string> labels;

    KMD5HashGenerator hashGenerator;
    labels.push_back(hashGenerator.GenerateHash(fContainer));
    labels.push_back(KResidualVector<ValueType>::Name());

    if (KEMFileInterface::GetInstance()->NumberWithLabels(labels)) {
        KEMFileInterface::GetInstance()->FindByLabels(fResidualVector, labels);
        solver.SetResidualVector(fResidualVector);
    }
}
}  // namespace KEMField

#endif /* KITERATIVESTATEREADER_DEF */
