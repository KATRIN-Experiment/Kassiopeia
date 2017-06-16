/*
 * KFMElectrostaticTypes.hh
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_UTILITY_INCLUDE_KFMELECTROSTATICTYPES_HH_
#define KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_UTILITY_INCLUDE_KFMELECTROSTATICTYPES_HH_


#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMDenseBoundaryIntegralMatrix.hh"

#include "KFMDenseBlockSparseMatrix.hh"
#include "KFMSparseBoundaryIntegralMatrix_BlockCompressedRow.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#include "KFMDenseBlockSparseMatrix_MPI.hh"
#include "KFMElectrostaticBoundaryIntegrator_MPI.hh"
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#endif



namespace KEMField {

namespace KFMElectrostaticTypes {

#ifdef KEMFIELD_USE_OPENCL
	#ifdef KEMFIELD_USE_MPI //mpi+opencl solver engines
		//#pragma message("Using MPI+OpenCL")

		typedef KFMElectrostaticBoundaryIntegrator_MPI<KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL>
		FastMultipoleEBI;

		typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
		FastMultipoleEBI, KFMDenseBlockSparseMatrix_MPI<FastMultipoleEBI::ValueType> >
		FastMultipoleSparseMatrix;

	#else //mpi not enabled, default to single threaded opencl
		//#pragma message("Using OpenCL only")

		typedef KFMElectrostaticBoundaryIntegrator<KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL>
		FastMultipoleEBI;

		typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
		FastMultipoleEBI, KFMDenseBlockSparseMatrix<FastMultipoleEBI::ValueType> >
		FastMultipoleSparseMatrix;

	#endif
#else //opencl not enabled, default to mpi/single threaded only
	#if KEMFIELD_USE_MPI
		//#pragma message("Using MPI Only")

		typedef KFMElectrostaticBoundaryIntegrator_MPI<KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
		FastMultipoleEBI;

		typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
		FastMultipoleEBI, KFMDenseBlockSparseMatrix_MPI<FastMultipoleEBI::ValueType> >
		FastMultipoleSparseMatrix;
	#else //nothing enabled, single threaded only
		//#pragma message("Using single thread")

		typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread>
		FastMultipoleEBI;

		typedef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow<KFMElectrostaticNodeObjects,
		FastMultipoleEBI, KFMDenseBlockSparseMatrix<FastMultipoleEBI::ValueType> >
		FastMultipoleSparseMatrix;
	#endif
#endif

typedef FastMultipoleEBI::ValueType ValueType;

typedef KFMDenseBoundaryIntegralMatrix<FastMultipoleEBI>
FastMultipoleDenseMatrix;

typedef KFMBoundaryIntegralMatrix< FastMultipoleDenseMatrix, FastMultipoleSparseMatrix>
FastMultipoleMatrix;

}

} /* namespace KEMField */



#endif /* KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_UTILITY_INCLUDE_KFMELECTROSTATICTYPES_HH_ */
