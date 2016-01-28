# header files
set( TRAJECTORIES_HEADER_BASENAMES
	KSTrajectoriesMessage.h

    KSTrajExactParticle.h
    KSTrajExactDerivative.h
    KSTrajExactError.h
    KSTrajExactTypes.h

    KSTrajAdiabaticParticle.h
    KSTrajAdiabaticDerivative.h
    KSTrajAdiabaticError.h
    KSTrajAdiabaticTypes.h

    KSTrajElectricParticle.h
    KSTrajElectricDerivative.h
    KSTrajElectricError.h
    KSTrajElectricTypes.h

    KSTrajMagneticParticle.h
    KSTrajMagneticDerivative.h
    KSTrajMagneticError.h
    KSTrajMagneticTypes.h

    KSTrajIntegratorRK54.h
    KSTrajIntegratorRKDP54.h
    KSTrajIntegratorRKDP853.h
    KSTrajIntegratorRK65.h
    KSTrajIntegratorRK86.h
    KSTrajIntegratorRK87.h
    KSTrajIntegratorRK8.h

    KSTrajInterpolatorFast.h
    KSTrajInterpolatorHermite.h
    KSTrajInterpolatorContinuousRungeKutta.h

    KSTrajTermPropagation.h
    KSTrajTermConstantForcePropagation.h
    KSTrajTermSynchrotron.h
    KSTrajTermGyration.h
    KSTrajTermDrift.h

    KSTrajControlCyclotron.h
    KSTrajControlTime.h
    KSTrajControlLength.h
    KSTrajControlEnergy.h
    KSTrajControlPositionNumericalError.h
    KSTrajControlMomentumNumericalError.h
    KSTrajControlMagneticMoment.h

    KSTrajTrajectoryExact.h
    KSTrajTrajectoryAdiabatic.h
    KSTrajTrajectoryElectric.h
    KSTrajTrajectoryMagnetic.h
    KSTrajTrajectoryLinear.h
)
set( TRAJECTORIES_HEADER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${TRAJECTORIES_HEADER_BASENAMES} )
	list( APPEND TRAJECTORIES_HEADER_FILENAMES ${TRAJECTORIES_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( TRAJECTORIES_SOURCE_BASENAMES
    KSTrajectoriesMessage.cxx

    KSTrajExactParticle.cxx
    KSTrajExactDerivative.cxx
    KSTrajExactError.cxx

    KSTrajAdiabaticParticle.cxx
    KSTrajAdiabaticDerivative.cxx
    KSTrajAdiabaticError.cxx

    KSTrajElectricParticle.cxx
    KSTrajElectricDerivative.cxx
    KSTrajElectricError.cxx

    KSTrajMagneticParticle.cxx
    KSTrajMagneticDerivative.cxx
    KSTrajMagneticError.cxx

    KSTrajIntegratorRK54.cxx
    KSTrajIntegratorRKDP54.cxx
    KSTrajIntegratorRKDP853.cxx
    KSTrajIntegratorRK65.cxx
    KSTrajIntegratorRK86.cxx
    KSTrajIntegratorRK87.cxx
    KSTrajIntegratorRK8.cxx

    KSTrajInterpolatorFast.cxx
    KSTrajInterpolatorHermite.cxx
    KSTrajInterpolatorContinuousRungeKutta.cxx

    KSTrajTermPropagation.cxx
    KSTrajTermConstantForcePropagation.cxx
    KSTrajTermSynchrotron.cxx
    KSTrajTermGyration.cxx
    KSTrajTermDrift.cxx

    KSTrajControlCyclotron.cxx
    KSTrajControlTime.cxx
    KSTrajControlLength.cxx
    KSTrajControlEnergy.cxx
    KSTrajControlPositionNumericalError.cxx
    KSTrajControlMomentumNumericalError.cxx
    KSTrajControlMagneticMoment.cxx

    KSTrajTrajectoryExact.cxx
    KSTrajTrajectoryAdiabatic.cxx
    KSTrajTrajectoryElectric.cxx
    KSTrajTrajectoryMagnetic.cxx
    KSTrajTrajectoryLinear.cxx
)
set( TRAJECTORIES_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${TRAJECTORIES_SOURCE_BASENAMES} )
	list( APPEND TRAJECTORIES_SOURCE_FILENAMES ${TRAJECTORIES_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( TRAJECTORIES_DATA_BASENAMES
)
set( TRAJECTORIES_DATA_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${TRAJECTORIES_DATA_BASENAMES} )
	list( APPEND TRAJECTORIES_DATA_FILENAMES ${TRAJECTORIES_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${TRAJECTORIES_HEADER_PATH} )
add_library( KassiopeiaTrajectories SHARED ${TRAJECTORIES_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaTrajectories PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaTrajectories
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
	KassiopeiaMath
)

# install
kasper_install_headers( ${TRAJECTORIES_HEADER_FILENAMES} )
kasper_install_data( ${TRAJECTORIES_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaTrajectories )
