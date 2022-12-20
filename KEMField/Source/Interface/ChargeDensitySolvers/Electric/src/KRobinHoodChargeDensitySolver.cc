/*
 * KRobinHoodChargeDensitySolver.cc
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#include "KRobinHoodChargeDensitySolver.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KElectrostaticBoundaryIntegrator.hh"
#include "KIterativeStateWriter.hh"
#include "KRobinHood.hh"
#include "KEMCoreMessage.hh"

#ifdef KEMFIELD_USE_VTK
#include "KVTKIterationPlotter.hh"
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KRobinHood_OpenCL.hh"
using KEMField::KRobinHood_OpenCL;
#endif

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
using KEMField::KMPIInterface;
#include "KRobinHood_MPI.hh"
using KEMField::KRobinHood_MPI;
#endif

#ifdef KEMFIELD_USE_OPENCL
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
using KEMField::KRobinHood_MPI_OpenCL;
#endif
#endif

#ifdef KEMFIELD_USE_MPI
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (true)
#endif
#endif

namespace KEMField
{

KRobinHoodChargeDensitySolver::KRobinHoodChargeDensitySolver() :
    fTolerance(1.e-8),
    fCheckSubInterval(100),
    fDisplayInterval(0),
    fWriteInterval(0),
    fPlotInterval(0),
    fCacheMatrixElements(false),
    fUseOpenCL(false),
    fUseVTK(false)
{}

KRobinHoodChargeDensitySolver::~KRobinHoodChargeDensitySolver()
{
#ifdef KEMFIELD_USE_OPENCL
    if (fUseOpenCL) {
        KOpenCLSurfaceContainer* oclContainer =
            dynamic_cast<KOpenCLSurfaceContainer*>(KOpenCLInterface::GetInstance()->GetActiveData());
        if (oclContainer)
            delete oclContainer;
        oclContainer = NULL;
        KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    }
#endif
}

void KRobinHoodChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: RobinHood solver got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    if (FindSolution(fTolerance, container) == false) {
        if (fUseOpenCL) {
#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
            //assign devices according to the number available and local process rank
            unsigned int proc_id = KMPIInterface::GetInstance()->GetProcess();
            int n_dev = KOpenCLInterface::GetInstance()->GetNumberOfDevices();
            int dev_id = proc_id % n_dev;  //fallback to global process rank if local is unavailable
           int local_rank = KMPIInterface::GetInstance()->GetLocalRank();
           if (local_rank != -1) {
               if (KMPIInterface::GetInstance()->SplitMode()) {
                   dev_id = (local_rank / 2) % n_dev;
               }
               else {
                   dev_id = (local_rank) % n_dev;
               }
           }
            KOpenCLInterface::GetInstance()->SetGPU(dev_id);
#endif

#ifdef KEMFIELD_USE_OPENCL
            KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(container);
            KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
            KOpenCLElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateOpenCLIntegrator(*oclContainer)};
            KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(*oclContainer, integrator);
            KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(*oclContainer, integrator);
            KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(*oclContainer, integrator);

#ifdef KEMFIELD_USE_MPI
            KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#else
            KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#endif
            robinHood.SetTolerance(fTolerance);
            robinHood.SetResidualCheckInterval(fCheckSubInterval);

            if (fDisplayInterval != 0) {
                MPI_SINGLE_PROCESS
                {
                    KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>* display =
                        new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>();
                    display->Interval(fDisplayInterval);
                    robinHood.AddVisitor(display);
                }
            }
            if (fWriteInterval != 0) {
                KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>* stateWriter =
                    new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(container);
                stateWriter->Interval(fWriteInterval);
                robinHood.AddVisitor(stateWriter);
            }
            if (fPlotInterval != 0) {
                if (fUseVTK == true) {
#ifdef KEMFIELD_USE_VTK
                    MPI_SINGLE_PROCESS
                    {
                        KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>* plotter =
                            new KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>();
                        plotter->Interval(fPlotInterval);
                        robinHood.AddVisitor(plotter);
                    }
#endif
                }
            }

            robinHood.Solve(A, x, b);

            MPI_SINGLE_PROCESS
            {
                SaveSolution(fTolerance, container);
            }
            return;
#endif
        }
        KElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateIntegrator()};
        KSquareMatrix<double>* A;
        if (fCacheMatrixElements)
            A = new KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator, true>(container, integrator);
        else
            A = new KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator>(container, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(container, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(container, integrator);

#ifdef KEMFIELD_USE_MPI
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI> robinHood;
#else
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
#endif
        robinHood.SetTolerance(fTolerance);
        robinHood.SetResidualCheckInterval(fCheckSubInterval);

        if (fDisplayInterval != 0) {
            MPI_SINGLE_PROCESS
            {
                auto* display = new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>();
                display->Interval(fDisplayInterval);
                robinHood.AddVisitor(display);
            }
        }
        if (fWriteInterval != 0) {
            MPI_SINGLE_PROCESS
            {
                auto* stateWriter = new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(container);
                stateWriter->Interval(fWriteInterval);
                robinHood.AddVisitor(stateWriter);
            }
        }
        if (fPlotInterval != 0) {
            if (fUseVTK == true) {
#ifdef KEMFIELD_USE_VTK
                MPI_SINGLE_PROCESS
                {
                    auto* plotter = new KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>();
                    plotter->Interval(fPlotInterval);
                    robinHood.AddVisitor(plotter);
                }
#endif
            }
        }

        robinHood.Solve(*A, x, b);

        MPI_SINGLE_PROCESS
        {
            SaveSolution(fTolerance, container);
        }
        return;
    }
}

void KRobinHoodChargeDensitySolver::SetSplitMode(bool choice)
{
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->SetSplitMode(choice);
#else
    (void)choice;  // fixes unused parameter warning
#endif
    return;
}


}  // namespace KEMField
