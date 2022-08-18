/*
 * KKrylovSolverFactory.hh
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVSOLVERFACTORY_HH_
#define KKRYLOVSOLVERFACTORY_HH_

#include "KBiconjugateGradientStabilized.hh"
#include "KEMSimpleException.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KIterativeKrylovSolver.hh"
#include "KIterativeSolverTimer.hh"
#include "KKrylovSolverConfiguration.hh"
#include "KMPIEnvironment.hh"
#include "KMatrixPreconditioner.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KSimpleIterativeKrylovSolver.hh"
#include "KSquareMatrix.hh"
#include "KTimeTerminator.hh"

#ifdef KEMFIELD_USE_VTK
#include "KVTKIterationPlotter.hh"
#include "KVTKResidualGraph.hh"
#endif

namespace KEMField
{

/** Create a new instance of the correct Krylov solver configured as
 * in the config given and without preconditioner or with the
 * preconditioner set to the given matrix. The returned solver class
 * may be different to the unpreconditioned case because the
 * unpreconditioned implementation is optimized and can not accept a
 * preconditioner. */
template<typename ValueType>
std::shared_ptr<KIterativeKrylovSolver<ValueType>>
KBuildKrylovSolver(const KKrylovSolverConfiguration& config,
                   std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                   std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner = nullptr);

/* private implementation class */

template<typename ValueType> class KKrylovSolverFactory
{

    friend std::shared_ptr<KIterativeKrylovSolver<ValueType>>
    KBuildKrylovSolver<ValueType>(const KKrylovSolverConfiguration& config,
                                  std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                                  std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner);

    KKrylovSolverFactory(const KKrylovSolverConfiguration& config,
                         std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                         std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner = nullptr);

    std::shared_ptr<KIterativeKrylovSolver<ValueType>> getSolver()
    {
        return fSolver;
    }

    void Build();

    void CreateStandard();
    void CreatePreconditioned();

    template<template<typename> class Trait> void CreateWithPreconditioner();

    void SetConfig();
    void SetMatrix();

    const KKrylovSolverConfiguration fConfig;
    const std::shared_ptr<const KSquareMatrix<ValueType>> fMatrix;
    const std::shared_ptr<const KSquareMatrix<ValueType>> fPreconditioner;
    std::shared_ptr<KIterativeKrylovSolver<ValueType>> fSolver;
};

template<typename ValueType>
std::shared_ptr<KIterativeKrylovSolver<ValueType>>
KBuildKrylovSolver(const KKrylovSolverConfiguration& config,
                   std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                   std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner)
{
    KKrylovSolverFactory<ValueType> factory(config, matrix, preconditioner);
    return factory.getSolver();
}

template<typename ValueType>
KKrylovSolverFactory<ValueType>::KKrylovSolverFactory(const KKrylovSolverConfiguration& config,
                                                      std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                                                      std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner) :
    fConfig(config),
    fMatrix(matrix),
    fPreconditioner(preconditioner)
{
    Build();
}

template<typename ValueType> void KKrylovSolverFactory<ValueType>::Build()
{
    if (! fPreconditioner)
        CreateStandard();
    else
        CreatePreconditioned();

    SetConfig();
    SetMatrix();
}

template<typename ValueType> void KKrylovSolverFactory<ValueType>::CreateStandard()
{
    std::string type = fConfig.GetSolverName();
    if (type == "gmres")
        fSolver = std::make_shared<KSimpleIterativeKrylovSolver<ValueType, KGeneralizedMinimalResidual>>();
    else if (type == "bicgstab")
        fSolver = std::make_shared<KSimpleIterativeKrylovSolver<ValueType, KBiconjugateGradientStabilized>>();
    else
        throw KEMSimpleException("Unknown solver type: " + type +
                                 ". Please ensure solver type is given in lower case.");
}

template<typename ValueType> void KKrylovSolverFactory<ValueType>::CreatePreconditioned()
{
    std::string type = fConfig.GetSolverName();
    if (type == "gmres")
        CreateWithPreconditioner<KPreconditionedGeneralizedMinimalResidual>();
    else if (type == "bicgstab")
        CreateWithPreconditioner<KPreconditionedBiconjugateGradientStabilized>();
    else
        throw KEMSimpleException("Unknown solver type. Please ensure solver type is given in lower case.");
}

template<typename ValueType>
template<template<typename> class Trait>
void KKrylovSolverFactory<ValueType>::CreateWithPreconditioner()
{
    auto solver = std::make_shared<KPreconditionedIterativeKrylovSolver<ValueType, Trait>>();

    auto transformedPrecon = std::make_shared<KMatrixPreconditioner<ValueType>>(fPreconditioner);

    solver->SetPreconditioner(transformedPrecon);
    fSolver = solver;
}

template<typename ValueType> void KKrylovSolverFactory<ValueType>::SetConfig()
{

    auto restartCond = std::make_shared<KIterativeKrylovRestartCondition>();

    restartCond->SetNumberOfIterationsBetweenRestart(fConfig.GetIterationsBetweenRestart());
    fSolver->SetRestartCondition(restartCond);

    fSolver->SetTolerance(fConfig.GetTolerance());
    fSolver->SetMaximumIterations(fConfig.GetMaxIterations());

    if (fConfig.IsUseTimer()) {
        fSolver->AddVisitor(
            new KTimeTerminator<ValueType>(fConfig.GetTimeLimitSeconds(), fConfig.GetStepsBetweenTimeChecks()));
    }

    if (fConfig.IsUseCheckpoints())
        throw KEMSimpleException("Multilevel preconditioned krylov solver does for now not support checkpoints");

    if (fConfig.IsUseTimer()) {
        throw KEMSimpleException("Multilevel preconditioned krylov solver does not"
                                 " support timed termination for now because it does not support"
                                 " checkpoints");
    }

    MPI_SINGLE_PROCESS
    {
        if (fConfig.IsUseCheckpoints()) {
            throw KEMSimpleException("Multilevel preconditioned krylov solver has"
                                     " checkpoint support not yet implemented here.");
        }


        if (fConfig.IsUseDisplay()) {
            fSolver->AddVisitor(new KIterationDisplay<ValueType>(std::string(fConfig.GetDisplayName())));
        }
#ifdef KEMFIELD_USE_VTK
        if (fConfig.IsUsePlot()) {
            fSolver->AddVisitor(new KVTKIterationPlotter<ValueType>());
        }
#endif

        if (fConfig.IsUseTimer()) {
            fSolver->AddVisitor(new KIterativeSolverTimer<ValueType>);
        }
    }
}

template<typename ValueType> void KKrylovSolverFactory<ValueType>::SetMatrix()
{
    fSolver->SetMatrix(fMatrix);
}

} /* namespace KEMField */


#endif /* KKRYLOVSOLVERFACTORY_HH_ */
