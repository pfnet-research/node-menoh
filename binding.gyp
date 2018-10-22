{
    "targets": [{
        "target_name": "menoh",
        "sources": [
            "src/menoh.cpp",
            "src/model.cpp"
        ],
        "include_dirs" : [
            "<!(node -e \"require('nan')\")"
        ],
        "conditions": [
            [ 'OS=="linux"', {
                "cflags_cc!": ["-fno-exceptions"],
                "cflags": [
                    "-std=c++11",
                    "-Wall",
                    "-g",
                    "-rdynamic"
                ],
            }],
            [ 'OS=="mac"', {
                "xcode_settings": {
                    "MACOSX_DEPLOYMENT_TARGET": "10.9",
                    "GCC_ENABLE_CPP_RTTI": "NO",
                    "GCC_ENABLE_CPP_EXCEPTIONS": "NO"
                    "CLANG_CXX_LIBRARY': 'libc++",
                    "CLANG_CXX_LANGUAGE_STANDARD": "c++11",
                    "GCC_VERSION": "com.apple.compilers.llvm.clang.1_0"
                },
            }],
            [ 'OS=="win"', {
                "variables": {
                    "prebuild_path%": "deps/win/menoh_v1.1.1",
                },
                "msvs_settings": {
                    "VCCLCompilerTool": {
                        "RuntimeTypeInfo": "false",
                        "ExceptionHandling": "1",
                        "DisableSpecificWarnings": [],
                    },
                },
                "include_dirs" : [
                    "<(prebuild_path)/include",
                ],
                "libraries": [ "../<(prebuild_path)/lib/menoh.lib" ]
            }, { # OS != "win",
                "cflags": [
                    "<!@(pkg-config --cflags menoh)"
                ],
                "ldflags": [
                    "<!@(pkg-config --libs-only-L menoh)",
                ],
                "libraries": [ "<!@(pkg-config --libs-only-l menoh)" ]
            }],
        ]
    }]
}
