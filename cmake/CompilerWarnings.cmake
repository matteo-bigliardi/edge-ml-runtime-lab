# A single place to decide what the compilers complain about, so the two
# toolchains this project builds on cannot drift apart silently.
#
# The warnings are attached PRIVATE: they are how *we* compile, not a
# constraint we impose on anything that links against the library.

function(tinytensor_set_warnings target)
    if(MSVC)
        set(warnings /W4 /permissive-)
        if(TINYTENSOR_WARNINGS_AS_ERRORS)
            list(APPEND warnings /WX)
        endif()
    else()
        # -Wconversion is the one that matters here: a tensor library is mostly
        # index arithmetic, and a silent int64/size_t narrowing is exactly the
        # bug class that surfaces as a wrong stride three layers down.
        set(warnings
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wnon-virtual-dtor
            -Wold-style-cast)
        if(TINYTENSOR_WARNINGS_AS_ERRORS)
            list(APPEND warnings -Werror)
        endif()
    endif()

    target_compile_options(${target} PRIVATE ${warnings})
endfunction()
