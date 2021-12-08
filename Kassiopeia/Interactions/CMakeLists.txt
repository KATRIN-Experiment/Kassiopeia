# header files
set( INTERACTIONS_HEADER_BASENAMES
    KSInteractionsMessage.h

    KSIntScattering.h

    KSIntSpinFlip.h
    KSIntSpinFlipPulse.h
    KSIntSpinRotateYPulse.h

    KSIntDecay.h
    KSIntDecayCalculator.h
    KSIntDecayCalculatorDeathConstRate.h
    KSIntDecayCalculatorGlukhovSpontaneous.h
    KSIntDecayCalculatorGlukhovDeExcitation.h
    KSIntDecayCalculatorGlukhovExcitation.h
    KSIntDecayCalculatorGlukhovIonisation.h
    KSIntDensity.h
    KSIntDensityConstant.h

    KSIntCalculator.h
    KSIntCalculatorConstant.h
    KSIntCalculatorHydrogen.h
    KSIntCalculatorIon.h
    KSIntCalculatorTritium.h
    KSIntCalculatorArgon.h

    KSIntSurfaceSpecular.h
    KSIntSurfaceUCN.h
    KSIntSurfaceDiffuse.h
    KSIntSurfaceMultiplication.h

    KSIntSurfaceSpinFlip.h

    KESSElasticElsepa.h
    KESSInelasticBetheFano.h
    KESSInelasticPenn.h
    KESSPhotoAbsorbtion.h
    KESSRelaxation.h
    KESSScatteringCalculator.h
    KESSSurfaceInteraction.h
    RydbergFerenc.h
    QuadGaussLegendre.h
    KSIntDecayCalculatorFerencIonisation.h
    KSIntDecayCalculatorFerencSpontaneous.h
    KSIntDecayCalculatorFerencBBRTransition.h
)
set( INTERACTIONS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${INTERACTIONS_HEADER_BASENAMES} )
    list( APPEND INTERACTIONS_HEADER_FILENAMES ${INTERACTIONS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( INTERACTIONS_SOURCE_BASENAMES
    KSInteractionsMessage.cxx

    KSIntScattering.cxx

    KSIntSpinFlip.cxx
    KSIntSpinFlipPulse.cxx
    KSIntSpinRotateYPulse.cxx

    KSIntDecay.cxx
    KSIntDecayCalculator.cxx
    KSIntDecayCalculatorDeathConstRate.cxx
    KSIntDecayCalculatorGlukhovSpontaneous.cxx
    KSIntDecayCalculatorGlukhovDeExcitation.cxx
    KSIntDecayCalculatorGlukhovExcitation.cxx
    KSIntDecayCalculatorGlukhovIonisation.cxx
    KSIntDensity.cxx
    KSIntDensityConstant.cxx

    KSIntCalculator.cxx
    KSIntCalculatorConstant.cxx
    KSIntCalculatorHydrogen.cxx
    KSIntCalculatorIon.cxx
    KSIntCalculatorTritium.cxx
    KSIntCalculatorArgon.cxx

    KSIntSurfaceSpecular.cxx
    KSIntSurfaceUCN.cxx
    KSIntSurfaceDiffuse.cxx
    KSIntSurfaceMultiplication.cxx

    KSIntSurfaceSpinFlip.cxx

    KESSElasticElsepa.cxx
    KESSInelasticBetheFano.cxx
    KESSInelasticPenn.cxx
    KESSPhotoAbsorbtion.cxx
    KESSRelaxation.cxx
    KESSScatteringCalculator.cxx
    KESSSurfaceInteraction.cxx
    RydbergFerenc.cxx
    KSIntDecayCalculatorFerencIonisation.cxx
    KSIntDecayCalculatorFerencSpontaneous.cxx
    KSIntDecayCalculatorFerencBBRTransition.cxx
)
set( INTERACTIONS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${INTERACTIONS_SOURCE_BASENAMES} )
    list( APPEND INTERACTIONS_SOURCE_FILENAMES ${INTERACTIONS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# data files
set( INTERACTIONS_DATA_BASENAMES
    Argon.txt
    argon_total_single_ionization_cross-section.txt
    argon_total_double_ionization_cross-section.txt
    argon_total_elastic_cross-section.txt
    argon_differential_elastic_cross-section.txt
    argon_excitation_state-1_cross-section.txt
    argon_excitation_state-2_cross-section.txt
    argon_excitation_state-3_cross-section.txt
    argon_excitation_state-4_cross-section.txt
    argon_excitation_state-5_cross-section.txt
    argon_excitation_state-6_cross-section.txt
    argon_excitation_state-7_cross-section.txt
    argon_excitation_state-8_cross-section.txt
    argon_excitation_state-9_cross-section.txt
    argon_excitation_state-10_cross-section.txt
    argon_excitation_state-11_cross-section.txt
    argon_excitation_state-12_cross-section.txt
    argon_excitation_state-13_cross-section.txt
    argon_excitation_state-14_cross-section.txt
    argon_excitation_state-15_cross-section.txt
    argon_excitation_state-16_cross-section.txt
    argon_excitation_state-17_cross-section.txt
    argon_excitation_state-18_cross-section.txt
    argon_excitation_state-19_cross-section.txt
    argon_excitation_state-20_cross-section.txt
    argon_excitation_state-21_cross-section.txt
    argon_excitation_state-22_cross-section.txt
    argon_excitation_state-23_cross-section.txt
    argon_excitation_state-24_cross-section.txt
    argon_excitation_state-25_cross-section.txt
    argon_differential_cross-section_excitation_state.10.txt
    argon_differential_cross-section_excitation_state.11.txt
    argon_differential_cross-section_excitation_state.12.txt
    argon_differential_cross-section_excitation_state.13.txt
    argon_differential_cross-section_excitation_state.14.txt
    argon_differential_cross-section_excitation_state.15.txt
    argon_differential_cross-section_excitation_state.16.txt
    argon_differential_cross-section_excitation_state.17.txt
    argon_differential_cross-section_excitation_state.18.txt
    argon_differential_cross-section_excitation_state.19.txt
    argon_differential_cross-section_excitation_state.1.txt
    argon_differential_cross-section_excitation_state.20.txt
    argon_differential_cross-section_excitation_state.21.txt
    argon_differential_cross-section_excitation_state.22.txt
    argon_differential_cross-section_excitation_state.23.txt
    argon_differential_cross-section_excitation_state.24.txt
    argon_differential_cross-section_excitation_state.25.txt
    argon_differential_cross-section_excitation_state.26.txt
    argon_differential_cross-section_excitation_state.27.txt
    argon_differential_cross-section_excitation_state.28.txt
    argon_differential_cross-section_excitation_state.29.txt
    argon_differential_cross-section_excitation_state.2.txt
    argon_differential_cross-section_excitation_state.30.txt
    argon_differential_cross-section_excitation_state.3.txt
    argon_differential_cross-section_excitation_state.4.txt
    argon_differential_cross-section_excitation_state.5.txt
    argon_differential_cross-section_excitation_state.6.txt
    argon_differential_cross-section_excitation_state.7.txt
    argon_differential_cross-section_excitation_state.8.txt
    argon_differential_cross-section_excitation_state.9.txt
    Hydrogen.txt
    Nitrogen.txt
    Tritium.txt
    Water.txt
    BetheFano.txt
    BetheFano_MFP.txt
    Penn.txt
    Penn_MFP.txt
    Elsepa.txt
    Elsepa_MFP.txt
    PhotoAbsorbtion.txt
)
set( INTERACTIONS_DATA_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Data
)
foreach( BASENAME ${INTERACTIONS_DATA_BASENAMES} )
    list( APPEND INTERACTIONS_DATA_FILENAMES ${INTERACTIONS_DATA_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaInteractions SHARED
    ${INTERACTIONS_SOURCE_FILENAMES} ${INTERACTIONS_HEADER_FILENAMES} )
target_include_directories(KassiopeiaInteractions
    PUBLIC $<BUILD_INTERFACE:${INTERACTIONS_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaInteractions
    PUBLIC
        KommonGsl
        KassiopeiaOperators
)

# install
kasper_install_headers( ${INTERACTIONS_HEADER_FILENAMES} )
kasper_install_data( ${INTERACTIONS_DATA_FILENAMES} )
kasper_install_libraries( KassiopeiaInteractions )
