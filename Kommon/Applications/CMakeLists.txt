
set( APPLICATION_NAMES
    Serializer
)

set( APPLICATION_LIBRARIES
    Kommon
)

foreach( NAME ${APPLICATION_NAMES} )
    add_executable( ${NAME} Source/${NAME}.cxx )
    target_link_libraries( ${NAME} ${APPLICATION_LIBRARIES}  )
    kasper_install_executables( ${NAME} )
endforeach()
