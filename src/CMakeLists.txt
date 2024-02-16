
set (control_sources
        #lems_vo.cpp
        Frame.cpp
        Dataset.cpp
        Pipeline.cpp
        utility.cpp
)

add_library( lems_vo_slam ${control_sources} )

#> Linking third-party libraries
target_link_libraries(lems_vo_slam
        ${THIRD_PARTY_LIBS})

