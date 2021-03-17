#ifndef KITERATIVESTATEWRITER_DEF
#define KITERATIVESTATEWRITER_DEF

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KIterativeSolver.hh"
#include "KMD5HashGenerator.hh"
#include "KMPIEnvironment.hh"
#include "KResidualVector.hh"
#include "KSurfaceContainer.hh"

#include <cstdio>
#include <deque>
#include <sstream>
#include <sys/stat.h>

namespace KEMField
{
struct KResidualThreshold
{
    KResidualThreshold() : fResidualThreshold(std::numeric_limits<double>::max()), fGeometryHash("") {}

    static std::string Name()
    {
        return "ResidualThreshold";
    }

    friend bool operator<(const KResidualThreshold& lhs, const KResidualThreshold& rhs)
    {
        return lhs.fResidualThreshold < rhs.fResidualThreshold;
    }

    double fResidualThreshold;
    std::string fGeometryHash;
};

template<typename Stream> Stream& operator>>(Stream& s, KResidualThreshold& r)
{
    s.PreStreamInAction(r);
    s >> r.fResidualThreshold;
    s >> r.fGeometryHash;
    s.PostStreamInAction(r);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KResidualThreshold& r)
{
    s.PreStreamOutAction(r);
    s << r.fResidualThreshold;
    s << r.fGeometryHash;
    s.PostStreamOutAction(r);
    return s;
}

template<typename ValueType> class KIterativeStateWriter : public KIterativeSolver<ValueType>::Visitor
{
  public:
    KIterativeStateWriter(KSurfaceContainer& container) :
        fContainer(container),
        fSaveNameRoot("partialConvergence"),
        fNConcurrentFiles(1)
    {}
    ~KIterativeStateWriter() override = default;

    void SaveNameRoot(const std::string& root)
    {
        fSaveNameRoot = root;
    }
    void NConcurrentFiles(unsigned int i)
    {
        if (i > 0)
            fNConcurrentFiles = i;
    }

    void Initialize(KIterativeSolver<ValueType>&) override;
    void Visit(KIterativeSolver<ValueType>&) override;
    void Finalize(KIterativeSolver<ValueType>&) override;

  private:
    KSurfaceContainer& fContainer;
    KResidualVector<ValueType> fResidualVector;
    std::string fSaveNameRoot;
    unsigned int fNConcurrentFiles;

    std::deque<std::string> fSavedContainerFiles;
    std::vector<std::string> fThresholdLabels;
};

template<typename ValueType> void KIterativeStateWriter<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
{
    MPI_SINGLE_PROCESS
    {
        fResidualVector.resize(solver.Dimension());

        KMD5HashGenerator hashGenerator;
        hashGenerator.Omit(KBasisTypes());
        fThresholdLabels.push_back(hashGenerator.GenerateHash(fContainer));
        fThresholdLabels.push_back(KResidualThreshold::Name());
    }
}

template<typename ValueType> void KIterativeStateWriter<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
{
    solver.CoalesceData();
    solver.GetResidualVector(fResidualVector);
    fResidualVector.SetIteration(solver.Iteration());

    MPI_SINGLE_PROCESS
    {
        KResidualThreshold residualThreshold;
        residualThreshold.fResidualThreshold = fResidualVector.InfinityNorm();

        KMD5HashGenerator hashGenerator;
        residualThreshold.fGeometryHash = hashGenerator.GenerateHash(fContainer);

        std::string activeNameRoot = KEMFileInterface::GetInstance()->GetActiveFileName().substr(
            0, KEMFileInterface::GetInstance()->GetActiveFileName().find_last_of("."));
        std::stringstream saveName;
        saveName << activeNameRoot << "_" << fSaveNameRoot << "_" << solver.Iteration()
                 << KEMFileInterface::GetInstance()->GetFileSuffix();

        KEMFileInterface::GetInstance()->Write(saveName.str(),
                                               residualThreshold,
                                               KResidualThreshold::Name(),
                                               fThresholdLabels);

        KEMFileInterface::GetInstance()->Write(saveName.str(), fContainer, KSurfaceContainer::Name());

        std::vector<std::string> vectorLabels;

        vectorLabels.push_back(residualThreshold.fGeometryHash);
        std::stringstream s;
        s << "iteration_" << solver.Iteration();
        vectorLabels.push_back(s.str());
        vectorLabels.push_back(KResidualVector<ValueType>::Name());
        KEMFileInterface::GetInstance()->Write(saveName.str(),
                                               fResidualVector,
                                               KResidualVector<ValueType>::Name(),
                                               vectorLabels);

        fSavedContainerFiles.push_back(saveName.str());

        if (fSavedContainerFiles.size() > fNConcurrentFiles) {
            std::remove(fSavedContainerFiles.front().c_str());
            fSavedContainerFiles.pop_front();
        }
    }
}

template<typename ValueType> void KIterativeStateWriter<ValueType>::Finalize(KIterativeSolver<ValueType>& solver)
{
    solver.GetResidualVector(fResidualVector);
    fResidualVector.SetIteration(solver.Iteration());

    MPI_SINGLE_PROCESS
    {
        KResidualThreshold residualThreshold;
        residualThreshold.fResidualThreshold = fResidualVector.InfinityNorm();

        KMD5HashGenerator hashGenerator;
        residualThreshold.fGeometryHash = hashGenerator.GenerateHash(fContainer);

        std::string activeNameRoot = KEMFileInterface::GetInstance()->GetActiveFileName().substr(
            0, KEMFileInterface::GetInstance()->GetActiveFileName().find_last_of("."));
        std::stringstream saveName;
        saveName << activeNameRoot << "_" << fSaveNameRoot << "_final"
                 << KEMFileInterface::GetInstance()->GetFileSuffix();

        KEMFileInterface::GetInstance()->Write(saveName.str(),
                                               residualThreshold,
                                               KResidualThreshold::Name(),
                                               fThresholdLabels);

        KEMFileInterface::GetInstance()->Write(saveName.str(), fContainer, KSurfaceContainer::Name());

        std::vector<std::string> vectorLabels;

        vectorLabels.push_back(residualThreshold.fGeometryHash);
        std::stringstream s;
        s << "iteration_" << solver.Iteration();
        vectorLabels.push_back(s.str());
        vectorLabels.push_back(KResidualVector<ValueType>::Name());
        KEMFileInterface::GetInstance()->Write(saveName.str(),
                                               fResidualVector,
                                               KResidualVector<ValueType>::Name(),
                                               vectorLabels);

        for (auto& file : fSavedContainerFiles)
            std::remove(file.c_str());
    }
}
}  // namespace KEMField


#endif /* KITERATIVESTATEWRITER_DEF */
