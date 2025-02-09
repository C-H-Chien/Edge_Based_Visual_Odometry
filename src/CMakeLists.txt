
set (control_sources
        Frame.cpp
        Dataset.cpp
        Pipeline.cpp
        utility.cpp
        MotionTracker.cpp
        toed/cpu_toed.cpp
)

add_library( lems_vo_slam ${control_sources} )

#> Linking third-party libraries
target_link_libraries(lems_vo_slam
        PUBLIC OpenMP::OpenMP_CXX
        ${THIRD_PARTY_LIBS})

