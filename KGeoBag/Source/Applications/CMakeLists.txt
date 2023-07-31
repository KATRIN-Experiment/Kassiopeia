option (KGeoBag_ENABLE_APP "Build KGeoBag applications" ON)
if (KGeoBag_ENABLE_APP)

    # names
    set( APPLICATIONS_NAMES
        KGeoBag
        RandomExtensionTester
        MetricsExtensionTester
        EulerAngles
        GeometryDump
    )

    # libraries
    set( APPLICATIONS_LIBRARIES
        KGeoBagBindings
        KGeoBagCore
        KGeoBagRandom
    )

    if( KGeoBag_USE_VTK )
        list( APPEND APPLICATIONS_NAMES
            GeometryViewer
            MeshViewer
            AxialMeshViewer
            PointTester
            NormalTester
            DistanceTester
            OutsideTester
            RandomPointTester
            MeshIntersectionTester
        )
        list( APPEND APPLICATIONS_LIBRARIES
            KGeoBagVtkVisualization
        )
    endif( KGeoBag_USE_VTK )

    if( KGeoBag_USE_ROOT )
        list( APPEND APPLICATIONS_NAMES
            GeometryViewerROOT
        )
        list( APPEND APPLICATIONS_LIBRARIES
            KGeoBagROOTVisualization
        )
    endif( KGeoBag_USE_ROOT )

    if( NOT BUILD_KSC )
        foreach( NAME ${APPLICATIONS_NAMES} )
            add_executable( ${NAME} Source/${NAME}.cc )
            target_link_libraries( ${NAME} ${APPLICATIONS_LIBRARIES} )
            kasper_install_executables( ${NAME} )
        endforeach()
    endif( NOT BUILD_KSC )

endif(KGeoBag_ENABLE_APP)
