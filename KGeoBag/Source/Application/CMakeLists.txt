# names
set( APPLICATION_NAMES
	KGeoBag
	RandomExtensionTester
	MetricsExtensionTester
    EulerAngles
)

# libraries
set( APPLICATION_LIBRARIES
    ${Kommon_LIBRARIES}
    KGeoBagMath
    KGeoBagCore
    KGeoBagShapes
    KGeoBagAppearance
    KGeoBagMesh
    KGeoBagAxialMesh
    KGeoBagBindings
    KGeoBagRandom
    KGeoBagMetrics
)
if( KGeoBag_USE_VTK )
	list( APPEND APPLICATION_NAMES
		GeometryViewer
		GeometryPrinter
		MeshViewer
		AxialMeshViewer
		PointTester
		NormalTester
		DistanceTester
		OutsideTester
		RandomPointTester
		MeshIntersectionTester
	)
	list( APPEND APPLICATION_LIBRARIES
	   ${Kommon_Vtk_LIBRARIES}
		KGeoBagVtkVisualization
	)
endif( KGeoBag_USE_VTK )

if( KGeoBag_USE_ROOT )
	list( APPEND APPLICATION_NAMES
		GeometryViewerROOT
	)
	list( APPEND APPLICATION_LIBRARIES
		KGeoBagROOTVisualization
	)
endif( KGeoBag_USE_ROOT )

# internal
kasper_internal_include_directories()

# target
if(APPLE)
	foreach( NAME ${APPLICATION_NAMES} )
		add_executable( ${NAME} MACOSX_BUNDLE Source/${NAME}.cc )
	endforeach()
else(APPLE)
	foreach( NAME ${APPLICATION_NAMES} )
		add_executable( ${NAME} Source/${NAME}.cc )
	endforeach()
endif(APPLE)

# install
foreach( NAME ${APPLICATION_NAMES} )
	target_link_libraries( ${NAME} ${APPLICATION_LIBRARIES} )
	kasper_install_executables( ${NAME} )
endforeach()
