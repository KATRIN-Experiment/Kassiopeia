kasper_install_config_subdir( Complete
    Complete/Objects.xml
    Complete/Geometry.xml
    Complete/Generators.xml
    Complete/Trajectories.xml
    Complete/Interactions.xml
    Complete/Navigators.xml
    Complete/Terminators.xml
    Complete/Writers.xml
    Complete/Fields.xml
)

kasper_install_config_subdir( Validation
    Validation/TestGenerator.xml
    Validation/TestTrajectory.xml
    Validation/TestSpaceInteraction.xml
    Validation/TestElectricFieldLines.xml
    Validation/TestField.xml
    Validation/TestRampedField.xml
    Validation/TestDiscreteRotationalField.xml
    Validation/TestNavigation.xml
    Validation/TestPotentialmap.xml
	Validation/TestSynchrotron.xml
)

if( Kassiopeia_USE_VTK )
    kasper_install_config_subdir( Examples
        Examples/VTK/AnalyticSimulation.xml
        Examples/VTK/QuadrupoleTrapSimulation.xml
        Examples/VTK/DipoleTrapSimulation.xml
        Examples/VTK/DipoleTrapMeshedSpaceSimulation.xml
        Examples/VTK/ToricTrapSimulation.xml
        Examples/VTK/PhotoMultiplierTubeSimulation.xml
    )
else( Kassiopeia_USE_VTK )
    kasper_install_config_subdir( Examples
        Examples/AnalyticSimulation.xml
        Examples/QuadrupoleTrapSimulation.xml
        Examples/DipoleTrapSimulation.xml
        Examples/DipoleTrapMeshedSpaceSimulation.xml
        Examples/ToricTrapSimulation.xml
        Examples/PhotoMultiplierTubeSimulation.xml
    )
endif( Kassiopeia_USE_VTK )

kasper_install_config_subdir( Examples
    Examples/Fields.xml
    Examples/Generators.xml
)
