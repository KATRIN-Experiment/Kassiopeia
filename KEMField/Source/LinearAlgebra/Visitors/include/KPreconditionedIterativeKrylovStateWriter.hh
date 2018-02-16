#ifndef __KPreconditionedIterativeKrylovStateWriter_H__
#define __KPreconditionedIterativeKrylovStateWriter_H__

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <set>

#include "KPreconditionedIterativeKrylovSolver.hh"

#include "KIterativeSolver.hh"

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KMD5HashGenerator.hh"

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
*@file KPreconditionedIterativeKrylovStateWriter.hh
*@class KPreconditionedIterativeKrylovStateWriter
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jan 13 23:12:06 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/



template <typename ValueType, template <typename> class ParallelTrait >
class KPreconditionedIterativeKrylovStateWriter: public KIterativeSolver<ValueType>::Visitor
{
    public:

        typedef KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait> SolverType;

        KPreconditionedIterativeKrylovStateWriter(std::vector< std::string > labels):
            KIterativeSolver<ValueType>::Visitor()
            {
                fLabels = labels;

                std::stringstream ss;
                ss << KEMFileInterface::GetInstance()->ActiveDirectory();
                ss << "/";
                ss << "Krylov_";
                fSaveNameRoot = ss.str();

                fPreviousStateFile = "";
            };

        virtual ~KPreconditionedIterativeKrylovStateWriter(){};

        virtual void Initialize(KIterativeSolver<ValueType>& solver)
        {
            KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* krylov_solver = NULL;
            krylov_solver = dynamic_cast< KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* >(&solver);

            if(krylov_solver != NULL)
            {
                krylov_solver->CoalesceData();
                ParallelTrait<ValueType>* trait = krylov_solver->GetTrait();

                if(trait != NULL)
                {
                    std::stringstream ss;
                    ss << fSaveNameRoot;
                    ss << trait->NameLabel();
                    fSaveNameRoot = ss.str();

                    fLabels.push_back(trait->NameLabel());
                    std::vector< std::string > stateLabels = fLabels;

                    #ifdef KEMFIELD_USE_MPI
                    std::stringstream mpi_label;
                    mpi_label << "_mpi_";
                    mpi_label << KMPIInterface::GetInstance()->GetProcess();
                    mpi_label << "_";
                    mpi_label << KMPIInterface::GetInstance()->GetNProcesses();
                    stateLabels.push_back(mpi_label.str());
                    #endif

                    //look to see if any previous solver states
                    //related to the current problem exist
                    //if one does, then we load its name into fPreviousStateFile
                    //so that when the next state is ready, we can delete it
                    std::set<std::string> prev_states = KEMFileInterface::GetInstance()->FileNamesWithLabels(stateLabels);
                    if(prev_states.size() == 1)
                    {
                        fPreviousStateFile = *(prev_states.begin());
                    }
                }
            }
        };

        virtual void Visit(KIterativeSolver<ValueType>& solver)
        {
            KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* krylov_solver = NULL;
            krylov_solver = dynamic_cast< KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* >(&solver);

            if(krylov_solver != NULL)
            {
                krylov_solver->CoalesceData();
                ParallelTrait<ValueType>* trait = krylov_solver->GetTrait();

                if(trait != NULL)
                {
                    //compute hash for the krylov solver state (for unique id, not reference)
                    KMD5HashGenerator hashGenerator;
                    hashGenerator.MaskedBits( 20 );
                    hashGenerator.Threshold( 1e-14 );
                    std::string unique_state_id = hashGenerator.GenerateHash(*trait);

                    std::vector< std::string > stateLabels;
                    stateLabels = fLabels;

                    std::stringstream saveName;
                    saveName << fSaveNameRoot;
                    saveName << "_";
                    saveName << unique_state_id.substr(0,6);
                    saveName << krylov_solver->Iteration();

                    #ifdef KEMFIELD_USE_MPI
                    std::stringstream mpi_label;
                    mpi_label << "_mpi_";
                    mpi_label << KMPIInterface::GetInstance()->GetProcess();
                    mpi_label << "_";
                    mpi_label << KMPIInterface::GetInstance()->GetNProcesses();
                    saveName << mpi_label.str();
                    stateLabels.push_back(mpi_label.str());
                    #endif

                    saveName << KEMFileInterface::GetInstance()->GetFileSuffix();

                    std::stringstream label_ss;
                    label_ss << "iteration_" <<krylov_solver->Iteration();
                    stateLabels.push_back(label_ss.str());
                    stateLabels.push_back(unique_state_id);

                    KEMFileInterface::GetInstance()->Write(saveName.str(),
                                                           *trait,
                                                           trait->NameLabel(),
                                                           stateLabels);

                    //remove old uneeded state
                    if(fPreviousStateFile != std::string(""))
                    {
                        std::remove(fPreviousStateFile.c_str());
                    }
                    fPreviousStateFile = saveName.str();

                }
            }
        }

        virtual void Finalize(KIterativeSolver<ValueType>& solver)
        {
            KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* krylov_solver = NULL;
            krylov_solver = dynamic_cast< KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>* >(&solver);


            if(krylov_solver != NULL)
            {
                krylov_solver->CoalesceData();
                ParallelTrait<ValueType>* trait = krylov_solver->GetTrait();

                if(trait != NULL)
                {
                    std::vector< std::string > stateLabels;
                    stateLabels = fLabels;

                    //compute hash for the krylov solver state (for unique id, not reference)
                    KMD5HashGenerator hashGenerator;
                    hashGenerator.MaskedBits( 20 );
                    hashGenerator.Threshold( 1e-14 );
                    std::string unique_state_id = hashGenerator.GenerateHash(*trait);

                    std::stringstream saveName;
                    saveName << fSaveNameRoot;
                    saveName << "_";
                    saveName << unique_state_id.substr(0,6);
                    saveName << krylov_solver->Iteration();

                    #ifdef KEMFIELD_USE_MPI
                    std::stringstream mpi_label;
                    mpi_label << "_mpi_";
                    mpi_label << KMPIInterface::GetInstance()->GetProcess();
                    mpi_label << "_";
                    mpi_label << KMPIInterface::GetInstance()->GetNProcesses();
                    saveName << mpi_label.str();
                    stateLabels.push_back(mpi_label.str());
                    #endif

                    saveName << "_final";
                    saveName << KEMFileInterface::GetInstance()->GetFileSuffix();

                    std::stringstream label_ss;
                    label_ss << "iteration_" <<krylov_solver->Iteration();
                    stateLabels.push_back(label_ss.str());
                    stateLabels.push_back("final");
                    stateLabels.push_back(unique_state_id);

                    KEMFileInterface::GetInstance()->Write(saveName.str(),
                                                           *trait,
                                                           trait->NameLabel(),
                                                           stateLabels);

                    //remove old uneeded state
                    if(fPreviousStateFile != std::string(""))
                    {
                        std::remove(fPreviousStateFile.c_str());
                    }

                }
            }
        };

    protected:

        std::vector< std::string > fLabels;
        std::string fUniqueID;
        std::string fSaveNameRoot;
        std::string fPreviousStateFile;

};



}


#endif /* __KPreconditionedIterativeKrylovStateWriter_H__ */
