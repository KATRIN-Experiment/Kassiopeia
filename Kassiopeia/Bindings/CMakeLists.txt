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
    # moved to KEMField

    # geometry
    Geometry/Include/KSGeoSpaceBuilder.h
    Geometry/Include/KSGeoSurfaceBuilder.h
    Geometry/Include/KSGeoSideBuilder.h

    # generators
    Generators/Include/KSGenValueFixBuilder.h
    Generators/Include/KSGenValueSetBuilder.h
    Generators/Include/KSGenValueListBuilder.h
    Generators/Include/KSGenValueUniformBuilder.h
    Generators/Include/KSGenValueBoltzmannBuilder.h
    Generators/Include/KSGenValueGaussBuilder.h
    Generators/Include/KSGenValueGeneralizedGaussBuilder.h
    Generators/Include/KSGenValueFermiBuilder.h
    Generators/Include/KSGenValueRadiusCylindricalBuilder.h
    Generators/Include/KSGenValueRadiusSphericalBuilder.h
    Generators/Include/KSGenValueRadiusFractionBuilder.h
    Generators/Include/KSGenValueAngleCosineBuilder.h
    Generators/Include/KSGenValueAngleSphericalBuilder.h
    Generators/Include/KSGenValueParetoBuilder.h
    Generators/Include/KSGenValueZFrustrumBuilder.h
    Generators/Include/KSGenEnergyCompositeBuilder.h
    Generators/Include/KSGenEnergyKryptonEventBuilder.h
    Generators/Include/KSGenEnergyRadonEventBuilder.h
    Generators/Include/KSGenEnergyLeadEventBuilder.h
    Generators/Include/KSGenEnergyBetaDecayBuilder.h
    Generators/Include/KSGenEnergyBetaRecoilBuilder.h
    Generators/Include/KSGenEnergyRydbergBuilder.h
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
    Generators/Include/KSGenPositionFrustrumCompositeBuilder.h
    Generators/Include/KSGenMomentumRectangularCompositeBuilder.h
    Generators/Include/KSGenDirectionSphericalCompositeBuilder.h
    Generators/Include/KSGenDirectionSurfaceCompositeBuilder.h
    Generators/Include/KSGenDirectionSphericalMagneticFieldBuilder.h
    Generators/Include/KSGenSpinCompositeBuilder.h
    Generators/Include/KSGenSpinRelativeCompositeBuilder.h
    Generators/Include/KSGenTimeCompositeBuilder.h
    Generators/Include/KSGenGeneratorCompositeBuilder.h
    Generators/Include/KSGenGeneratorTextFileBuilder.h

    # trajectories
    Trajectories/Include/KSTrajIntegratorRK54Builder.h
    Trajectories/Include/KSTrajIntegratorRKDP54Builder.h
    Trajectories/Include/KSTrajIntegratorRKDP853Builder.h
    Trajectories/Include/KSTrajIntegratorRK65Builder.h
    Trajectories/Include/KSTrajIntegratorRK86Builder.h
    Trajectories/Include/KSTrajIntegratorRK87Builder.h
    Trajectories/Include/KSTrajIntegratorRK8Builder.h
    Trajectories/Include/KSTrajIntegratorSym4Builder.h
    Trajectories/Include/KSTrajInterpolatorFastBuilder.h
    Trajectories/Include/KSTrajInterpolatorHermiteBuilder.h
    Trajectories/Include/KSTrajInterpolatorContinuousRungeKuttaBuilder.h
    Trajectories/Include/KSTrajTermPropagationBuilder.h
    Trajectories/Include/KSTrajTermConstantForcePropagationBuilder.h
    Trajectories/Include/KSTrajTermSynchrotronBuilder.h
    Trajectories/Include/KSTrajTermGyrationBuilder.h
    Trajectories/Include/KSTrajTermGravityBuilder.h
    Trajectories/Include/KSTrajTermDriftBuilder.h
    Trajectories/Include/KSTrajControlTimeBuilder.h
    Trajectories/Include/KSTrajControlLengthBuilder.h
    Trajectories/Include/KSTrajControlEnergyBuilder.h
    Trajectories/Include/KSTrajControlPositionNumericalErrorBuilder.h
    Trajectories/Include/KSTrajControlMomentumNumericalErrorBuilder.h
    Trajectories/Include/KSTrajControlMagneticMomentBuilder.h
    Trajectories/Include/KSTrajControlCyclotronBuilder.h
    Trajectories/Include/KSTrajControlBChangeBuilder.h
    Trajectories/Include/KSTrajControlSpinPrecessionBuilder.h
    Trajectories/Include/KSTrajControlMDotBuilder.h
    Trajectories/Include/KSTrajTrajectoryExactBuilder.h
    Trajectories/Include/KSTrajTrajectoryExactSpinBuilder.h
    Trajectories/Include/KSTrajTrajectoryAdiabaticSpinBuilder.h
    Trajectories/Include/KSTrajTrajectoryAdiabaticBuilder.h
    Trajectories/Include/KSTrajTrajectoryExactTrappedBuilder.h
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
    Interactions/Include/KSIntCalculatorIonBuilder.h
    Interactions/Include/KSIntCalculatorArgonBuilder.h
    Interactions/Include/KSIntCalculatorKESSBuilder.h
    Interactions/Include/KSIntScatteringBuilder.h
    Interactions/Include/KSIntSpinFlipBuilder.h
    Interactions/Include/KSIntSpinFlipPulseBuilder.h
    Interactions/Include/KSIntSpinRotateYPulseBuilder.h
    Interactions/Include/KSIntDecayBuilder.h
    Interactions/Include/KSIntSurfaceSpecularBuilder.h
    Interactions/Include/KSIntSurfaceUCNBuilder.h
    Interactions/Include/KSIntSurfaceDiffuseBuilder.h
    Interactions/Include/KSIntSurfaceMultiplicationBuilder.h
    Interactions/Include/KSIntSurfaceSpinFlipBuilder.h
    Interactions/Include/KESSSurfaceInteractionBuilder.h
    Interactions/Include/KESSElasticElsepaBuilder.h
    Interactions/Include/KESSInelasticPennBuilder.h
    Interactions/Include/KESSInelasticBetheFanoBuilder.h

    # modifiers
    Modifiers/Include/KSModDynamicEnhancementBuilder.h
    Modifiers/Include/KSModSplitOnTurnBuilder.h
    Modifiers/Include/KSModEventReportBuilder.h

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
    Terminators/Include/KSTermXRangeBuilder.h
    Terminators/Include/KSTermYRangeBuilder.h
    Terminators/Include/KSTermZRangeBuilder.h
    Terminators/Include/KSTermMinEnergyBuilder.h
    Terminators/Include/KSTermMaxEnergyBuilder.h
    Terminators/Include/KSTermMinLongEnergyBuilder.h
    Terminators/Include/KSTermMaxLongEnergyBuilder.h
    Terminators/Include/KSTermMaxTimeBuilder.h
    Terminators/Include/KSTermMaxTotalTimeBuilder.h
    Terminators/Include/KSTermMaxStepTimeBuilder.h
    Terminators/Include/KSTermMaxLengthBuilder.h
    Terminators/Include/KSTermMaxStepsBuilder.h
    Terminators/Include/KSTermTrappedBuilder.h
    Terminators/Include/KSTermSecondariesBuilder.h
    Terminators/Include/KSTermOutputBuilder.h
    Terminators/Include/KSTermMinDistanceBuilder.h
    Terminators/Include/KSTermStepsizeBuilder.h
    Terminators/Include/KSTermMagnetronBuilder.h
    Terminators/Include/KSTermZHRadiusBuilder.h

    # writers
    Writers/Include/KSWriteASCIIBuilder.h

    # visualization

    # simulation
    Simulation/Include/KSRootMagneticFieldBuilder.h
    Simulation/Include/KSRootElectricFieldBuilder.h
    Simulation/Include/KSRootGeneratorBuilder.h
    Simulation/Include/KSRootTrajectoryBuilder.h
    Simulation/Include/KSRootSpaceInteractionBuilder.h
    Simulation/Include/KSRootSurfaceInteractionBuilder.h
    Simulation/Include/KSRootSpaceNavigatorBuilder.h
    Simulation/Include/KSRootStepModifierBuilder.h
    Simulation/Include/KSRootTrackModifierBuilder.h
    Simulation/Include/KSRootEventModifierBuilder.h
    Simulation/Include/KSRootRunModifierBuilder.h
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
        Writers/Include/KSWriteROOTConditionOutputBuilder.h
        Writers/Include/KSWriteROOTConditionPeriodicBuilder.h
        Writers/Include/KSWriteROOTConditionTerminatorBuilder.h
        Writers/Include/KSWriteROOTConditionStepBuilder.h
        Visualization/Include/KSROOTTrackPainterBuilder.h
        Visualization/Include/KSROOTPotentialPainterBuilder.h
        Visualization/Include/KSROOTMagFieldPainterBuilder.h
        Visualization/Include/KSROOTZonalHarmonicsPainterBuilder.h
    )

    if( Kassiopeia_USE_VTK )
        list( APPEND BINDINGS_HEADER_FILES
            Visualization/Include/KSVTKGeneratorPainterBuilder.h
            Visualization/Include/KSVTKTrackPainterBuilder.h
            Visualization/Include/KSVTKTrackTerminatorPainterBuilder.h
        )
    endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

if( Kassiopeia_USE_VTK )
    list( APPEND BINDINGS_HEADER_FILES
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
    # moved to KEMField. This is for backward XML compatibility
    Fields/Source/KSFieldKEMFieldObjectsBuilder.cxx

    # geometry
    Geometry/Source/KSGeoSpaceBuilder.cxx
    Geometry/Source/KSGeoSurfaceBuilder.cxx
    Geometry/Source/KSGeoSideBuilder.cxx

    # generators
    Generators/Source/KSGenValueFixBuilder.cxx
    Generators/Source/KSGenValueSetBuilder.cxx
    Generators/Source/KSGenValueListBuilder.cxx
    Generators/Source/KSGenValueUniformBuilder.cxx
    Generators/Source/KSGenValueBoltzmannBuilder.cxx
    Generators/Source/KSGenValueGaussBuilder.cxx
    Generators/Source/KSGenValueGeneralizedGaussBuilder.cxx
    Generators/Source/KSGenValueFermiBuilder.cxx
    Generators/Source/KSGenValueRadiusCylindricalBuilder.cxx
    Generators/Source/KSGenValueRadiusSphericalBuilder.cxx
    Generators/Source/KSGenValueRadiusFractionBuilder.cxx
    Generators/Source/KSGenValueAngleCosineBuilder.cxx
    Generators/Source/KSGenValueAngleSphericalBuilder.cxx
    Generators/Source/KSGenValueParetoBuilder.cxx
    Generators/Source/KSGenValueZFrustrumBuilder.cxx
    Generators/Source/KSGenEnergyCompositeBuilder.cxx
    Generators/Source/KSGenEnergyKryptonEventBuilder.cxx
    Generators/Source/KSGenEnergyRadonEventBuilder.cxx
    Generators/Source/KSGenEnergyLeadEventBuilder.cxx
    Generators/Source/KSGenEnergyBetaDecayBuilder.cxx
    Generators/Source/KSGenEnergyBetaRecoilBuilder.cxx
    Generators/Source/KSGenEnergyRydbergBuilder.cxx
    Generators/Source/KSGenNCompositeBuilder.cxx
    Generators/Source/KSGenLCompositeBuilder.cxx
    Generators/Source/KSGenLStatisticalBuilder.cxx
    Generators/Source/KSGenLUniformMaxNBuilder.cxx
    Generators/Source/KSGenPositionSurfaceAdjustmentStepBuilder.cxx
    Generators/Source/KSGenPositionRectangularCompositeBuilder.cxx
    Generators/Source/KSGenPositionCylindricalCompositeBuilder.cxx
    Generators/Source/KSGenPositionSphericalCompositeBuilder.cxx
    Generators/Source/KSGenPositionFrustrumCompositeBuilder.cxx
    Generators/Source/KSGenPositionSpaceRandomBuilder.cxx
    Generators/Source/KSGenPositionSurfaceRandomBuilder.cxx
    Generators/Source/KSGenPositionMeshSurfaceRandomBuilder.cxx
    Generators/Source/KSGenPositionMaskBuilder.cxx
    Generators/Source/KSGenPositionFluxTubeBuilder.cxx
    Generators/Source/KSGenPositionHomogeneousFluxTubeBuilder.cxx
    Generators/Source/KSGenMomentumRectangularCompositeBuilder.cxx
    Generators/Source/KSGenDirectionSphericalCompositeBuilder.cxx
    Generators/Source/KSGenDirectionSurfaceCompositeBuilder.cxx
    Generators/Source/KSGenDirectionSphericalMagneticFieldBuilder.cxx
    Generators/Source/KSGenSpinCompositeBuilder.cxx
    Generators/Source/KSGenSpinRelativeCompositeBuilder.cxx
    Generators/Source/KSGenTimeCompositeBuilder.cxx
    Generators/Source/KSGenGeneratorCompositeBuilder.cxx
    Generators/Source/KSGenGeneratorTextFileBuilder.cxx

    # trajectories
    Trajectories/Source/KSTrajIntegratorRK54Builder.cxx
    Trajectories/Source/KSTrajIntegratorRKDP54Builder.cxx
    Trajectories/Source/KSTrajIntegratorRKDP853Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK65Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK86Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK87Builder.cxx
    Trajectories/Source/KSTrajIntegratorRK8Builder.cxx
    Trajectories/Source/KSTrajIntegratorSym4Builder.cxx
    Trajectories/Source/KSTrajInterpolatorFastBuilder.cxx
    Trajectories/Source/KSTrajInterpolatorHermiteBuilder.cxx
    Trajectories/Source/KSTrajInterpolatorContinuousRungeKuttaBuilder.cxx
    Trajectories/Source/KSTrajTermPropagationBuilder.cxx
    Trajectories/Source/KSTrajTermConstantForcePropagationBuilder.cxx
    Trajectories/Source/KSTrajTermSynchrotronBuilder.cxx
    Trajectories/Source/KSTrajTermGyrationBuilder.cxx
    Trajectories/Source/KSTrajTermGravityBuilder.cxx
    Trajectories/Source/KSTrajTermDriftBuilder.cxx
    Trajectories/Source/KSTrajControlCyclotronBuilder.cxx
    Trajectories/Source/KSTrajControlBChangeBuilder.cxx
    Trajectories/Source/KSTrajControlSpinPrecessionBuilder.cxx
    Trajectories/Source/KSTrajControlMDotBuilder.cxx
    Trajectories/Source/KSTrajControlTimeBuilder.cxx
    Trajectories/Source/KSTrajControlLengthBuilder.cxx
    Trajectories/Source/KSTrajControlEnergyBuilder.cxx
    Trajectories/Source/KSTrajControlPositionNumericalErrorBuilder.cxx
    Trajectories/Source/KSTrajControlMomentumNumericalErrorBuilder.cxx
    Trajectories/Source/KSTrajControlMagneticMomentBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryExactBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryExactSpinBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryAdiabaticSpinBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryAdiabaticBuilder.cxx
    Trajectories/Source/KSTrajTrajectoryExactTrappedBuilder.cxx
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
    Interactions/Source/KSIntCalculatorIonBuilder.cxx
    Interactions/Source/KSIntCalculatorArgonBuilder.cxx
    Interactions/Source/KSIntCalculatorKESSBuilder.cxx
    Interactions/Source/KSIntScatteringBuilder.cxx
    Interactions/Source/KSIntSpinFlipBuilder.cxx
    Interactions/Source/KSIntSpinFlipPulseBuilder.cxx
    Interactions/Source/KSIntSpinRotateYPulseBuilder.cxx
    Interactions/Source/KSIntDecayBuilder.cxx
    Interactions/Source/KSIntSurfaceSpecularBuilder.cxx
    Interactions/Source/KSIntSurfaceUCNBuilder.cxx
    Interactions/Source/KSIntSurfaceDiffuseBuilder.cxx
    Interactions/Source/KSIntSurfaceMultiplicationBuilder.cxx
    Interactions/Source/KSIntSurfaceSpinFlipBuilder.cxx
    Interactions/Source/KESSSurfaceInteractionBuilder.cxx
    Interactions/Source/KESSElasticElsepaBuilder.cxx
    Interactions/Source/KESSInelasticPennBuilder.cxx
    Interactions/Source/KESSInelasticBetheFanoBuilder.cxx

    # modifiers
    Modifiers/Source/KSModDynamicEnhancementBuilder.cxx
    Modifiers/Source/KSModSplitOnTurnBuilder.cxx
    Modifiers/Source/KSModEventReportBuilder.cxx

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
    Terminators/Source/KSTermXRangeBuilder.cxx
    Terminators/Source/KSTermYRangeBuilder.cxx
    Terminators/Source/KSTermZRangeBuilder.cxx
    Terminators/Source/KSTermMinEnergyBuilder.cxx
    Terminators/Source/KSTermMaxEnergyBuilder.cxx
    Terminators/Source/KSTermMinLongEnergyBuilder.cxx
    Terminators/Source/KSTermMaxLongEnergyBuilder.cxx
    Terminators/Source/KSTermMaxTimeBuilder.cxx
    Terminators/Source/KSTermMaxTotalTimeBuilder.cxx
    Terminators/Source/KSTermMaxStepTimeBuilder.cxx
    Terminators/Source/KSTermMaxLengthBuilder.cxx
    Terminators/Source/KSTermMaxStepsBuilder.cxx
    Terminators/Source/KSTermTrappedBuilder.cxx
    Terminators/Source/KSTermSecondariesBuilder.cxx
    Terminators/Source/KSTermOutputBuilder.cxx
    Terminators/Source/KSTermMinDistanceBuilder.cxx
    Terminators/Source/KSTermStepsizeBuilder.cxx
    Terminators/Source/KSTermMagnetronBuilder.cxx
    Terminators/Source/KSTermZHRadiusBuilder.cxx

    # writers
    Writers/Source/KSWriteASCIIBuilder.cxx

    # visualization

    # simulation
    Simulation/Source/KSRootMagneticFieldBuilder.cxx
    Simulation/Source/KSRootElectricFieldBuilder.cxx
    Simulation/Source/KSRootGeneratorBuilder.cxx
    Simulation/Source/KSRootTrajectoryBuilder.cxx
    Simulation/Source/KSRootSpaceInteractionBuilder.cxx
    Simulation/Source/KSRootSurfaceInteractionBuilder.cxx
    Simulation/Source/KSRootSpaceNavigatorBuilder.cxx
    Simulation/Source/KSRootStepModifierBuilder.cxx
    Simulation/Source/KSRootTrackModifierBuilder.cxx
    Simulation/Source/KSRootEventModifierBuilder.cxx
    Simulation/Source/KSRootRunModifierBuilder.cxx
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
        Writers/Source/KSWriteROOTConditionOutputBuilder.cxx
        Writers/Source/KSWriteROOTConditionPeriodicBuilder.cxx
        Writers/Source/KSWriteROOTConditionTerminatorBuilder.cxx
        Writers/Source/KSWriteROOTConditionStepBuilder.cxx
        Visualization/Source/KSROOTTrackPainterBuilder.cxx
        Visualization/Source/KSROOTPotentialPainterBuilder.cxx
        Visualization/Source/KSROOTMagFieldPainterBuilder.cxx
        Visualization/Source/KSROOTZonalHarmonicsPainterBuilder.cxx
    )
    if( Kassiopeia_USE_VTK )
        list( APPEND BINDINGS_SOURCE_FILES
            Visualization/Source/KSVTKGeneratorPainterBuilder.cxx
            Visualization/Source/KSVTKTrackPainterBuilder.cxx
            Visualization/Source/KSVTKTrackTerminatorPainterBuilder.cxx
        )
    endif( Kassiopeia_USE_VTK )
endif( Kassiopeia_USE_ROOT )

if( Kassiopeia_USE_VTK )
    list( APPEND BINDINGS_SOURCE_FILES
        Writers/Source/KSWriteVTKBuilder.cxx
    )
endif( Kassiopeia_USE_VTK )

# target
add_library( KassiopeiaBindings SHARED
    ${BINDINGS_SOURCE_FILES} ${BINDINGS_HEADER_FILES}
)

add_library( KassiopeiaBindingsHeaders INTERFACE )

# get header paths from collected header files
foreach(HEADER ${BINDINGS_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KassiopeiaBindings PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
    target_include_directories(KassiopeiaBindingsHeaders INTERFACE $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)

target_include_directories(KassiopeiaBindings PUBLIC $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaBindings
    PUBLIC
        KommonCore
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
    PRIVATE
        KEMFieldBindings
)

# install
kasper_install_headers( ${BINDINGS_HEADER_FILES} )
kasper_install_libraries( KassiopeiaBindings KassiopeiaBindingsHeaders )
