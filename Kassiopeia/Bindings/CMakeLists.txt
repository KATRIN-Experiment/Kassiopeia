# headers
set( BINDINGS_HEADER_FILES
    # objects
    Objects/Include/KSCommandMemberBuilder.h
    Objects/Include/KSCommandMemberSimpleBuilder.h
    Objects/Include/KSComponentGroupBuilder.h
    Objects/Include/KSComponentMemberBuilder.h
    Objects/Include/KSComponentDeltaBuilder.h
    Objects/Include/KSComponentIntegralBuilder.h
    Objects/Include/KSComponentMaximumBuilder.h
    Objects/Include/KSComponentMinimumBuilder.h
    Objects/Include/KSComponentMaximumAtBuilder.h
    Objects/Include/KSComponentMinimumAtBuilder.h

    # fields
    Fields/Include/KSFieldElectromagnetBuilder.h
    Fields/Include/KSFieldElectrostaticBuilder.h
    Fields/Include/KSFieldKrylovBEMSolverBuilder.h
    Fields/Include/KSFieldFMElectrostaticParametersBuilder.h
    Fields/Include/KSFieldMagneticConstantBuilder.h
    Fields/Include/KSFieldMagneticDipoleBuilder.h
    Fields/Include/KSFieldMagneticSuperPositionBuilder.h
    Fields/Include/KSFieldElectricConstantBuilder.h
    Fields/Include/KSFieldElectricQuadrupoleBuilder.h
    Fields/Include/KSFieldMagneticRampedBuilder.h
    Fields/Include/KSFieldElectricRampedBuilder.h

    # geometry
    Geometry/Include/KSGeoSpaceBuilder.h
    Geometry/Include/KSGeoSurfaceBuilder.h
    Geometry/Include/KSGeoSideBuilder.h

    # generators
    Generators/Include/KSGenValueFixBuilder.h
    Generators/Include/KSGenValueSetBuilder.h
    Generators/Include/KSGenValueListBuilder.h
    Generators/Include/KSGenValueUniformBuilder.h
    Generators/Include/KSGenValueGaussBuilder.h
    Generators/Include/KSGenValueRadiusCylindricalBuilder.h
    Generators/Include/KSGenValueRadiusSphericalBuilder.h
    Generators/Include/KSGenValueAngleSphericalBuilder.h
    Generators/Include/KSGenValueParetoBuilder.h
    Generators/Include/KSGenEnergyCompositeBuilder.h
    Generators/Include/KSGenEnergyKryptonEventBuilder.h
    Generators/Include/KSGenEnergyRadonEventBuilder.h
    Generators/Include/KSGenEnergyLeadEventBuilder.h
    Generators/Include/KSGenEnergyBetaDecayBuilder.h
    Generators/Include/KSGenNCompositeBuilder.h
    Generators/Include/KSGenLCompositeBuilder.h
    Generators/Include/KSGenLStatisticalBuilder.h
    Generators/Include/KSGenLUniformMaxNBuilder.h
    Generators/Include/KSGenPositionSurfaceAdjustmentStepBuilder.h
    Generators/Include/KSGenPositionRectangularCompositeBuilder.h
    Generators/Include/KSGenPositionCylindricalCompositeBuilder.h
    Generators/Include/KSGenPositionSphericalCompositeBuilder.h
    Generators/Include/KSGenPositionSpaceRandomBuilder.h
    Generators/Include/KSGenPositionSurfaceRandomBuilder.h
    Generators/Include/KSGenPositionMeshSurfaceRandomBuilder.h
    Generators/Include/KSGenPositionMaskBuilder.h
    Generators/Include/KSGenPositionFluxTubeBuilder.h
    Generators/Include/KSGenPositionHomogeneousFluxTubeBuilder.h
    Generators/Include/KSGenMomentumRectangularCompositeBuilder.h
    Generators/Include/KSGenDirectionSphericalCompositeBuilder.h
    Generators/Include/KSGenDirectionSurfaceCompositeBuilder.h
    Generators/Include/KSGenTimeCompositeBuilder.h
    Generators/Include/KSGenGeneratorCompositeBuilder.h


    # trajectories
    Trajectories/Include/KSTrajIntegratorRK54Builder.h
    Trajectories/Include/KSTrajIntegratorRKDP54Builder.h
    Trajectories/Include/KSTrajIntegratorRKDP853Builder.h
    Trajectories/Include/KSTrajIntegratorRK65Builder.h
    Trajectories/Include/KSTrajIntegratorRK86Builder.h
    Trajectories/Include/KSTrajIntegratorRK87Builder.h
    Trajectories/Include/KSTrajIntegratorRK8Builder.h
    Trajectories/Include/KSTrajInterpolatorFastBuilder.h
    Trajectories/Include/KSTrajInterpolatorHermiteBuilder.h
    Trajectories/Include/KSTrajInterpolatorContinuousRungeKuttaBuilder.h
    Trajectories/Include/KSTrajTermPropagationBuilder.h
    Trajectories/Include/KSTrajTermConstantForcePropagationBuilder.h
    Trajectories/Include/KSTrajTermSynchrotronBuilder.h
    Trajectories/Include/KSTrajTermGyrationBuilder.h
    Trajectories/Include/KSTrajTermDriftBuilder.h
    Trajectories/Include/KSTrajControlTimeBuilder.h
    Trajectories/Include/KSTrajControlLengthBuilder.h
    Trajectories/Include/KSTrajControlEnergyBuilder.h
    Trajectories/Include/KSTrajControlPositionNumericalErrorBuilder.h
    Trajectories/Include/KSTrajControlMomentumNumericalErrorBuilder.h
    Trajectories/Include/KSTrajControlMagneticMomentBuilder.h
    Trajectories/Include/KSTrajControlCyclotronBuilder.h
    Trajectories/Include/KSTrajTrajectoryExactBuilder.h
    Trajectories/Include/KSTrajTrajectoryAdiabaticBuilder.h
    Trajectories/Include/KSTrajTrajectoryElectricBuilder.h
    Trajectories/Include/KSTrajTrajectoryMagneticBuilder.h
    Trajectories/Include/KSTrajTrajectoryLinearBuilder.h

    # interactions
    Interactions/Include/KSIntDensityConstantBuilder.h
    Interactions/Include/KSIntCalculatorConstantBuilder.h
    Interactions/Include/KSIntDecayCalculatorDeathConstRateBuilder.h
    Interactions/Include/KSIntDecayCalculatorGlukhovSpontaneousBuilder.h
    Interactions/Include/KSIntDecayCalculatorGlukhovDeExcitationBuilder.h
    Interactions/Include/KSIntDecayCalculatorGlukhovExcitationBuilder.h
    Interactions/Include/KSIntDecayCalculatorGlukhovIonisationBuilder.h
    Interactions/Include/KSIntDecayCalculatorFerencSpontaneousBuilder.h
    Interactions/Include/KSIntDecayCalculatorFerencBBRTransitionBuilder.h
    Interactions/Include/KSIntDecayCalculatorFerencIonisationBuilder.h
    Interactions/Include/KSIntCalculatorHydrogenBuilder.h
    Interactions/Include/KSIntCalculatorArgonBuilder.h
    Interactions/Include/KSIntScatteringBuilder.h
    Interactions/Include/KSIntDecayBuilder.h
    Interactions/Include/KSIntSurfaceSpecularBuilder.h
    Interactions/Include/KSIntSurfaceDiffuseBuilder.h
    Interactions/Include/KSIntSurfaceMultiplicationBuilder.h
    Interactions/Include/KESSSurfaceInteractionBuilder.h
    Interactions/Include/KESSElasticElsepaBuilder.h
    Interactions/Include/KESSInelasticPennBuilder.h
    Interactions/Include/KESSInelasticBetheFanoBuilder.h

    # modifiers
    Modifiers/Include/KSModDynamicEnhancementBuilder.h

    # navigators
    Navigators/Include/KSNavSurfaceBuilder.h
    Navigators/Include/KSNavSpaceBuilder.h
    Navigators/Include/KSNavMeshedSpaceBuilder.h

    # terminators
    Terminators/Include/KSTermDeathBuilder.h
    Terminators/Include/KSTermMinZBuilder.h
    Terminators/Include/KSTermMaxZBuilder.h
    Terminators/Include/KSTermMinRBuilder.h
    Terminators/Include/KSTermMaxRBuilder.h
    Terminators/Include/KSTermMinEnergyBuilder.h
    Terminators/Include/KSTermMaxEnergyBuilder.h
    Terminators/Include/KSTermMinLongEnergyBuilder.h
    Terminators/Include/KSTermMaxLongEnergyBuilder.h
    Terminators/Include/KSTermMaxTimeBuilder.h
    Terminators/Include/KSTermMaxLengthBuilder.h
    Terminators/Include/KSTermMaxStepsBuilder.h
    Terminators/Include/KSTermTrappedBuilder.h
    Terminators/Include/KSTermSecondariesBuilder.h
    Terminators/Include/KSTermOutputBuilder.h
    Terminators/Include/KSTermMinDistanceBuilder.h
    Terminators/Include/KSTermStepsizeBuilder.h

    # writers
    Writers/Include/KSWriteASCIIBuilder.h

    # visualization

    # simulation
    Simulation/Include/KSRootMagneticFieldBuilder.h
    Simulation/Include/KSRootElectricFieldBuilder.h
    Simulation/Include/KSRootEventModifierBuilder.h
    Simulation/Include/KSRootGeneratorBuilder.h
    Simulation/Include/KSRootTrajectoryBuilder.h
    Simulation/Include/KSRootSpaceInteractionBuilder.h
    Simulation/Include/KSRootSurfaceInteractionBuilder.h
    Simulation/Include/KSRootSpaceNavigatorBuilder.h
    Simulation/Include/KSRootStepModifierBuilder.h
    Simulation/Include/KSRootSurfaceNavigatorBuilder.h
    Simulation/Include/KSRootTerminatorBuilder.h
    Simulation/Include/KSRootWriterBuilder.h
    Simulation/Include/KSRootBuilder.h
    Simulation/Include/KSSimulationBuilder.h
)

if( Kassiopeia_USE_ROOT )
    list( APPEND BINDINGS_HEADER_FILES
        Objects/Include/KSComponentMathBuilder.h
        Generators/Include/KSGenValueFormulaBuilder.h
        Generators/Include/KSGenValueHistogramBuilder.h
        Generators/Include/KSGenGeneratorSimulationBuilder.h
        Writers/Include/KSWriteROOTBuilder.h
        Visualization/Include/KSROOTTrackPainterBuilder.h
        Visualization/Include/KSROOTPotentialPainterBuilder.h
        Visualization/Include/KSROOTMagFieldPainterBuilder.h
    )
    if( Kassiopeia_USE_VTK )
        list( APPEND BINDINGS_HEADER_FILES
            Visualization/Include/KSVTKTrackPainterBuilder.h
            Visualization/Include/KSVTKTrackTerminatorPainterBuilder.h
        )
        endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

if( Kassiopeia_USE_VTK )
    list( APPEND BINDINGS_HEADER_FILES
        Fields/Include/KSFieldElectricPotentialmapBuilder.h
        Writers/Include/KSWriteVTKBuilder.h
    )
endif( Kassiopeia_USE_VTK )

# sources
set( BINDINGS_SOURCE_FILES
    # objects
    Objects/Source/KSCommandMemberBuilder.cxx
    Objects/Source/KSCommandMemberSimpleBuilder.cxx
    Objects/Source/KSComponentGroupBuilder.cxx
    Objects/Source/KSComponentMemberBuilder.cxx
    Objects/Source/KSComponentDeltaBuilder.cxx
    Objects/Source/KSComponentIntegralBuilder.cxx
    Objects/Source/KSComponentMaximumBuilder.cxx
    Objects/Source/KSComponentMinimumBuilder.cxx
    Objects/Source/KSComponentMaximumAtBuilder.cxx
    Objects/Source/KSComponentMinimumAtBuilder.cxx

    # fields
    Fields/Source/KSFieldElectromagnetBuilder.cxx
    Fields/Source/KSFieldElectrostaticBuilder.cxx
    Fields/Source/KSFieldKrylovBEMSolverBuilder.cxx
    Fields/Source/KSFieldFMElectrostaticParametersBuilder.cxx
    Fields/Source/KSFieldMagneticConstantBuilder.cxx
    Fields/Source/KSFieldMagneticDipoleBuilder.cxx
    Fields/Source/KSFieldMagneticSuperPositionBuilder.cxx
    Fields/Source/KSFieldElectricConstantBuilder.cxx
    Fields/Source/KSFieldElectricQuadrupoleBuilder.cxx
    Fields/Source/KSFieldMagneticRampedBuilder.cxx
    Fields/Source/KSFieldElectricRampedBuilder.cxx

    # geometry
    Geometry/Source/KSGeoSpaceBuilder.cxx
    Geometry/Source/KSGeoSurfaceBuilder.cxx
    Geometry/Source/KSGeoSideBuilder.cxx

    # generators
    Generators/Source/KSGenValueFixBuilder.cxx
    Generators/Source/KSGenValueSetBuilder.cxx
    Generators/Source/KSGenValueListBuilder.cxx
    Generators/Source/KSGenValueUniformBuilder.cxx
    Generators/Source/KSGenValueGaussBuilder.cxx
    Generators/Source/KSGenValueRadiusCylindricalBuilder.cxx
    Generators/Source/KSGenValueRadiusSphericalBuilder.cxx
    Generators/Source/KSGenValueAngleSphericalBuilder.cxx
    Generators/Source/KSGenValueParetoBuilder.cxx
    Generators/Source/KSGenEnergyCompositeBuilder.cxx
    Generators/Source/KSGenEnergyKryptonEventBuilder.cxx
    Generators/Source/KSGenEnergyRadonEventBuilder.cxx
    Generators/Source/KSGenEnergyLeadEventBuilder.cxx
    Generators/Source/KSGenEnergyBetaDecayBuilder.cxx
    Generators/Source/KSGenNCompositeBuilder.cxx
    Generators/Source/KSGenLCompositeBuilder.cxx
    Generators/Source/KSGenLStatisticalBuilder.cxx
    Generators/Source/KSGenLUniformMaxNBuilder.cxx
    Generators/Source/KSGenPositionSurfaceAdjustmentStepBuilder.cxx
    Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx
    Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx
    Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx
    Generators/Source/KSGenPositionSpaceRandomBuilder.cxx
    Generators/Source/KSGenPositionSurfaceRandomBuilder.cxx
    Generators/Source/KSGenPositionMeshSurfaceRandomBuilder.cxx
    Generators/Source/KSGenPositionMaskBuilder.cxx
    Generators/Source/KSGenPositionFluxTubeBuilder.cxx
    Generators/Source/KSGenPositionHomogeneousFluxTubeBuilder.cxx
    Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx
    Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx
    Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx
    Generators/Source/KSGenTimeCompositeBuilder.cxx
    Generators/Source/KSGenGeneratorCompositeBuilder.cxx

    # trajectories
    Trajectories/Source/KSTrajIntegratorRK54Builder.cxx
    Trajectories/Source/KSTrajIntegratorRKDP54Builder.cxx
    Trajectories/Source/KSTrajIntegratorRKDP853Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK65Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK86Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK87Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK8Builder.cxx
    Trajectories/Source/KSTrajInterpolatorFastBuilder.cxx
    Trajectories/Source/KSTrajInterpolatorHermiteBuilder.cxx
    Trajectories/Source/KSTrajInterpolatorContinuousRungeKuttaBuilder.cxx
    Trajectories/Source/KSTrajTermPropagationBuilder.cxx
    Trajectories/Source/KSTrajTermConstantForcePropagationBuilder.cxx
    Trajectories/Source/KSTrajTermSynchrotronBuilder.cxx
    Trajectories/Source/KSTrajTermGyrationBuilder.cxx
    Trajectories/Source/KSTrajTermDriftBuilder.cxx
    Trajectories/Source/KSTrajControlCyclotronBuilder.cxx
    Trajectories/Source/KSTrajControlTimeBuilder.cxx
    Trajectories/Source/KSTrajControlLengthBuilder.cxx
    Trajectories/Source/KSTrajControlEnergyBuilder.cxx
    Trajectories/Source/KSTrajControlPositionNumericalErrorBuilder.cxx
    Trajectories/Source/KSTrajControlMomentumNumericalErrorBuilder.cxx
    Trajectories/Source/KSTrajControlMagneticMomentBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryElectricBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryMagneticBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryLinearBuilder.cxx

    # interactions
    Interactions/Source/KSIntDensityConstantBuilder.cxx
    Interactions/Source/KSIntCalculatorConstantBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorDeathConstRateBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorGlukhovSpontaneousBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorGlukhovExcitationBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorGlukhovDeExcitationBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorGlukhovIonisationBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorFerencSpontaneousBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorFerencBBRTransitionBuilder.cxx
    Interactions/Source/KSIntDecayCalculatorFerencIonisationBuilder.cxx
    Interactions/Source/KSIntCalculatorHydrogenBuilder.cxx
    Interactions/Source/KSIntCalculatorArgonBuilder.cxx
    Interactions/Source/KSIntScatteringBuilder.cxx
    Interactions/Source/KSIntDecayBuilder.cxx
    Interactions/Source/KSIntSurfaceSpecularBuilder.cxx
    Interactions/Source/KSIntSurfaceDiffuseBuilder.cxx
    Interactions/Source/KSIntSurfaceMultiplicationBuilder.cxx
    Interactions/Source/KESSSurfaceInteractionBuilder.cxx
    Interactions/Source/KESSElasticElsepaBuilder.cxx
    Interactions/Source/KESSInelasticPennBuilder.cxx
    Interactions/Source/KESSInelasticBetheFanoBuilder.cxx

    # modifiers
    Modifiers/Source/KSModDynamicEnhancementBuilder.cxx

    # navigators
    Navigators/Source/KSNavSurfaceBuilder.cxx
    Navigators/Source/KSNavSpaceBuilder.cxx
    Navigators/Source/KSNavMeshedSpaceBuilder.cxx

    # terminators
    Terminators/Source/KSTermDeathBuilder.cxx
    Terminators/Source/KSTermMinZBuilder.cxx
    Terminators/Source/KSTermMaxZBuilder.cxx
    Terminators/Source/KSTermMinRBuilder.cxx
    Terminators/Source/KSTermMaxRBuilder.cxx
    Terminators/Source/KSTermMinEnergyBuilder.cxx
    Terminators/Source/KSTermMaxEnergyBuilder.cxx
    Terminators/Source/KSTermMinLongEnergyBuilder.cxx
    Terminators/Source/KSTermMaxLongEnergyBuilder.cxx
    Terminators/Source/KSTermMaxTimeBuilder.cxx
    Terminators/Source/KSTermMaxLengthBuilder.cxx
    Terminators/Source/KSTermMaxStepsBuilder.cxx
    Terminators/Source/KSTermTrappedBuilder.cxx
    Terminators/Source/KSTermSecondariesBuilder.cxx
    Terminators/Source/KSTermOutputBuilder.cxx
    Terminators/Source/KSTermMinDistanceBuilder.cxx
    Terminators/Source/KSTermStepsizeBuilder.cxx


    # writers
    Writers/Source/KSWriteASCIIBuilder.cxx

    # visualization

    # simulation
    Simulation/Source/KSRootMagneticFieldBuilder.cxx
    Simulation/Source/KSRootElectricFieldBuilder.cxx
    Simulation/Source/KSRootEventModifierBuilder.cxx
    Simulation/Source/KSRootGeneratorBuilder.cxx
    Simulation/Source/KSRootTrajectoryBuilder.cxx
    Simulation/Source/KSRootSpaceInteractionBuilder.cxx
    Simulation/Source/KSRootSurfaceInteractionBuilder.cxx
    Simulation/Source/KSRootSpaceNavigatorBuilder.cxx
    Simulation/Source/KSRootStepModifierBuilder.cxx
    Simulation/Source/KSRootSurfaceNavigatorBuilder.cxx
    Simulation/Source/KSRootTerminatorBuilder.cxx
    Simulation/Source/KSRootWriterBuilder.cxx
    Simulation/Source/KSRootBuilder.cxx
    Simulation/Source/KSSimulationBuilder.cxx
)

if( Kassiopeia_USE_ROOT )
    list( APPEND BINDINGS_SOURCE_FILES
        Objects/Source/KSComponentMathBuilder.cxx
        Generators/Source/KSGenValueFormulaBuilder.cxx
        Generators/Source/KSGenValueHistogramBuilder.cxx
        Generators/Source/KSGenGeneratorSimulationBuilder.cxx
        Writers/Source/KSWriteROOTBuilder.cxx
        Visualization/Source/KSROOTTrackPainterBuilder.cxx
        Visualization/Source/KSROOTPotentialPainterBuilder.cxx
        Visualization/Source/KSROOTMagFieldPainterBuilder.cxx
    )
    if( Kassiopeia_USE_VTK )
        list( APPEND BINDINGS_SOURCE_FILES
            Visualization/Source/KSVTKTrackPainterBuilder.cxx
            Visualization/Source/KSVTKTrackTerminatorPainterBuilder.cxx
        )
    endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

if( Kassiopeia_USE_VTK )
    list( APPEND BINDINGS_SOURCE_FILES
        Fields/Source/KSFieldElectricPotentialmapBuilder.cxx
        Writers/Source/KSWriteVTKBuilder.cxx
    )
endif( Kassiopeia_USE_VTK )

# libraries
set( BINDINGS_LIBRARIES
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    ${KEMField_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaFields
    KassiopeiaGenerators
    KassiopeiaGeometry
    KassiopeiaTrajectories
    KassiopeiaInteractions
    KassiopeiaModifiers
    KassiopeiaNavigators
    KassiopeiaTerminators
    KassiopeiaWriters
    KassiopeiaVisualization
    KassiopeiaSimulation
)

# internal
kasper_internal_include_directories(
    Objects/Include
    Operators/Include
    Fields/Include
    Generators/Include
    Geometry/Include
    Trajectories/Include
    Interactions/Include
    Modifiers/Include
    Navigators/Include
    Terminators/Include
    Writers/Include
    Visualization/Include
    Simulation/Include
)

# target
add_library( KassiopeiaBindings SHARED
    ${BINDINGS_SOURCE_FILES}
)
set_target_properties( KassiopeiaBindings PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaBindings
    ${BINDINGS_LIBRARIES}
)

# install
kasper_install_headers( ${BINDINGS_HEADER_FILES} )
kasper_install_libraries( KassiopeiaBindings )
