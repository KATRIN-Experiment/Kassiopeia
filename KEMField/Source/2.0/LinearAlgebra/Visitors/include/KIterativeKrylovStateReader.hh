#ifndef __KIterativeKrylovStateReader_H__
#define __KIterativeKrylovStateReader_H__

#include <string>
#include <sstream>

#include "KIterativeKrylovSolver.hh"
#include "KIterativeSolver.hh"

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess()==0)
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
*@file KIterativeKrylovStateReader.hh
*@class KIterativeKrylovStateReader
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 14 15:20:19 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


template <typename ValueType, template <typename> class ParallelTrait, template <typename> class ParallelTraitState >
class KIterativeKrylovStateReader: public KIterativeSolver<ValueType>::Visitor
{
    public:

        typedef KSimpleIterativeKrylovSolver<ValueType, ParallelTrait> SolverType;


        KIterativeKrylovStateReader(std::vector< std::string > labels):
            KIterativeSolver<ValueType>::Visitor()
            {
                fVerbosity = 0;
                fLabels = labels;
            };

        virtual ~KIterativeKrylovStateReader(){};

        void SetVerbosity(int v){fVerbosity = v;}

        virtual void Initialize(KIterativeSolver<ValueType>& solver)
        {
            KSimpleIterativeKrylovSolver<ValueType, ParallelTrait>* krylov_solver = NULL;
            krylov_solver = dynamic_cast< KSimpleIterativeKrylovSolver<ValueType, ParallelTrait>* >(&solver);

            if(krylov_solver != NULL)
            {
                ParallelTrait<ValueType>* trait = krylov_solver->GetTrait();

                if(trait != NULL)
                {
                    ParallelTraitState<ValueType> trait_state;
                    unsigned int n_states = 0;

                    std::vector< std::string > stateLabels = fLabels;
                    stateLabels.push_back(trait->NameLabel());

                    #ifdef KEMFIELD_USE_MPI
                    std::stringstream mpi_label;
                    mpi_label << "_mpi_";
                    mpi_label << KMPIInterface::GetInstance()->GetProcess();
                    mpi_label << "_";
                    mpi_label << KMPIInterface::GetInstance()->GetNProcesses();
                    stateLabels.push_back(mpi_label.str());
                    #endif

                    n_states = KEMFileInterface::GetInstance()->NumberWithLabels(stateLabels);
                    if(n_states == 1)
                    {
                        KEMFileInterface::GetInstance()->FindByLabels(trait_state, stateLabels);
                        MPI_SINGLE_PROCESS
                        {
                            if(fVerbosity > 2)
                            {
                                KEMField::cout<<"KPreconditionedIterativeKrylovSolver::Initalize: Found previously saved ";
                                KEMField::cout<<trait->Name()<<" Krylov space state. ";
                            }
                        }
                    }

                    if(n_states == 1)
                    {
                        trait_state.SynchronizeData();
                        trait->SetState(trait_state);
                        MPI_SINGLE_PROCESS
                        {
                            if(fVerbosity > 2)
                            {
                                KEMField::cout<<"Done loading."<<KEMField::endl;
                            }
                        }
                    };

                }
            }
        };

        virtual void Visit(KIterativeSolver<ValueType>&){};
        virtual void Finalize(KIterativeSolver<ValueType>&){};

    protected:

        int fVerbosity;
        std::vector<std::string> fLabels;

};


}

#endif /* __KIterativeKrylovStateReader_H__ */
