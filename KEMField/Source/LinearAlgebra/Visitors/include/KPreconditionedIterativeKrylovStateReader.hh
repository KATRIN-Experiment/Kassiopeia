#ifndef __KPreconditionedIterativeKrylovStateReader_H__
#define __KPreconditionedIterativeKrylovStateReader_H__

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KIterativeSolver.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"

#include <sstream>
#include <string>

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS
#endif
#endif


namespace KEMField
{

/**
*
*@file KPreconditionedIterativeKrylovStateReader.hh
*@class KPreconditionedIterativeKrylovStateReader
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 14 15:20:19 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ValueType, template<typename> class ParallelTrait, template<typename> class ParallelTraitState>
class KPreconditionedIterativeKrylovStateReader : public KIterativeSolver<ValueType>::Visitor
{
  public:
    typedef KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait> SolverType;

    KPreconditionedIterativeKrylovStateReader(std::vector<std::string> labels) : KIterativeSolver<ValueType>::Visitor()
    {
        fLabels = labels;
        fVerbosity = 0;
    };

    ~KPreconditionedIterativeKrylovStateReader() override{};

    void SetVerbosity(int v)
    {
        fVerbosity = v;
    }

    void Initialize(KIterativeSolver<ValueType>& solver) override
    {
        KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* krylov_solver = nullptr;
        krylov_solver = dynamic_cast<KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>*>(&solver);

        if (krylov_solver != nullptr) {
            ParallelTrait<ValueType>* trait = krylov_solver->GetTrait();

            if (trait != nullptr) {
                ParallelTraitState<ValueType> trait_state;
                unsigned int n_states = 0;

                std::vector<std::string> stateLabels = fLabels;

#ifdef KEMFIELD_USE_MPI
                std::stringstream mpi_label;
                mpi_label << "_mpi_";
                mpi_label << KMPIInterface::GetInstance()->GetProcess();
                mpi_label << "_";
                mpi_label << KMPIInterface::GetInstance()->GetNProcesses();
                stateLabels.push_back(mpi_label.str());
#endif

                stateLabels.push_back(trait->NameLabel());
                n_states = KEMFileInterface::GetInstance()->NumberWithLabels(stateLabels);
                if (n_states == 1) {
                    KEMFileInterface::GetInstance()->FindByLabels(trait_state, stateLabels);

                    MPI_SINGLE_PROCESS
                    {
                        if (fVerbosity > 2) {
                            KEMField::cout
                                << "KPreconditionedIterativeKrylovSolver::Initalize: Found previously saved ";
                            KEMField::cout << trait->Name() << " Krylov space state. ";
                        }
                    }
                }

                if (n_states == 1) {
                    trait_state.SynchronizeData();
                    trait->SetState(trait_state);

                    MPI_SINGLE_PROCESS
                    {
                        if (fVerbosity > 2) {
                            KEMField::cout << "Done loading." << KEMField::endl;
                        }
                    }
                }
            }
        }
    };

    void Visit(KIterativeSolver<ValueType>&) override{};
    void Finalize(KIterativeSolver<ValueType>&) override{};

  protected:
    int fVerbosity;
    std::vector<std::string> fLabels;
};


}  // namespace KEMField

#endif /* __KPreconditionedIterativeKrylovStateReader_H__ */
