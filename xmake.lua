set_project("optinum")
set_version("0.0.8")
set_xmakever("2.7.0")

-- Set C++ standard
set_languages("c++20")

-- Add build options
add_rules("mode.debug", "mode.release")

-- Compiler warnings and flags
add_cxxflags("-Wall", "-Wextra", "-Wpedantic")
add_cxxflags("-Wno-reorder", "-Wno-narrowing", "-Wno-array-bounds")
add_cxxflags("-Wno-unused-variable", "-Wno-unused-parameter", "-Wno-stringop-overflow", "-Wno-unused-but-set-variable")
add_cxxflags("-Wno-gnu-line-marker")

-- Enable AVX/AVX2 optimizations for SIMD (use explicit flags for Nix compatibility)
add_cxxflags("-mavx", "-mavx2", "-mfma")

-- Add global search paths for packages in ~/.local
local home = os.getenv("HOME")
if home then
    add_includedirs(path.join(home, ".local/include"))
    add_linkdirs(path.join(home, ".local/lib"))
end

-- Add devbox/nix paths for system packages
local cmake_prefix = os.getenv("CMAKE_PREFIX_PATH")
if cmake_prefix then
    add_includedirs(path.join(cmake_prefix, "include"))
    add_linkdirs(path.join(cmake_prefix, "lib"))
end

local pkg_config = os.getenv("PKG_CONFIG_PATH")
if pkg_config then
    -- Split PKG_CONFIG_PATH by ':' and process each path
    for _, pkgconfig_path in ipairs(pkg_config:split(':')) do
        if os.isdir(pkgconfig_path) then
            -- PKG_CONFIG_PATH typically points to .../lib/pkgconfig
            -- We want to get the prefix (two levels up) to find include and lib
            local lib_dir = path.directory(pkgconfig_path)  -- .../lib
            local prefix_dir = path.directory(lib_dir)      -- .../
            local include_dir = path.join(prefix_dir, "include")

            -- Avoid pulling in host headers/libs (breaks hermetic Nix builds by
            -- mixing /usr headers with Nix's glibc headers).
            if prefix_dir == "/usr" or prefix_dir == "/usr/local" then
                goto continue
            end

            if os.isdir(lib_dir) then
                add_linkdirs(lib_dir)
            end
            if os.isdir(include_dir) then
                add_includedirs(include_dir)
            end
        end
        ::continue::
    end
end

-- Options
option("examples")
    set_default(false)
    set_showmenu(true)
    set_description("Build examples")
option_end()

option("tests")
    set_default(false)
    set_showmenu(true)
    set_description("Enable tests")
option_end()

option("short_namespace")
    set_default(false)
    set_showmenu(true)
    set_description("Enable short namespace alias (on)")
option_end()

option("expose_all")
    set_default(false)
    set_showmenu(true)
    set_description("Expose all submodule functions in optinum:: namespace")
option_end()

-- Define datapod package (from git)
package("datapod")
    set_sourcedir(path.join(os.projectdir(), "build/_deps/datapod-src"))

    on_fetch(function (package)
        local sourcedir = package:sourcedir()
        if not os.isdir(sourcedir) then
            print("Fetching datapod from git...")
            os.mkdir(path.directory(sourcedir))
            os.execv("git", {"clone", "--quiet", "--depth", "1", "--branch", "0.0.15",
                            "-c", "advice.detachedHead=false",
                            "https://github.com/robolibs/datapod.git", sourcedir})
        end
    end)

    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"))
        import("package.tools.cmake").install(package, configs, {cmake_generator = "Unix Makefiles"})
    end)
package_end()

-- Add required packages
add_requires("datapod")

if has_config("tests") then
    add_requires("doctest")
end

-- Main library target
target("optinum")
    set_kind("static")

    -- Add source files
    add_files("src/optinum/**.cpp")

    -- Add header files
    add_headerfiles("include/(optinum/**.hpp)")
    add_includedirs("include", {public = true})

    -- Link dependencies
    add_packages("datapod")

    -- Add SHORT_NAMESPACE define if enabled
    if has_config("short_namespace") then
        add_defines("SHORT_NAMESPACE", {public = true})
    end

    -- Add OPTINUM_EXPOSE_ALL define if enabled
    if has_config("expose_all") then
        add_defines("OPTINUM_EXPOSE_ALL", {public = true})
    end

    -- Set install files
    add_installfiles("include/(optinum/**.hpp)")
    on_install(function (target)
        local installdir = target:installdir()
        os.cp(target:targetfile(), path.join(installdir, "lib", path.filename(target:targetfile())))
    end)
target_end()

-- Examples (only build when optinum is the main project)
if has_config("examples") and os.projectdir() == os.curdir() then
    for _, filepath in ipairs(os.files("examples/*.cpp")) do
        local filename = path.basename(filepath)
        target(filename)
            set_kind("binary")
            add_files(filepath)
            add_deps("optinum")
            add_packages("datapod")

            -- Always enable SHORT_NAMESPACE for examples
            add_defines("SHORT_NAMESPACE")

            add_includedirs("include")
        target_end()
    end
end

-- Tests (only build when optinum is the main project)
if has_config("tests") and os.projectdir() == os.curdir() then
    for _, filepath in ipairs(os.files("test/**.cpp")) do
        local filename = path.basename(filepath)
        target(filename)
            set_kind("binary")
            add_files(filepath)
            add_deps("optinum")
            add_packages("datapod", "doctest")
            add_includedirs("include")
            add_defines("DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN")

            -- Always enable SHORT_NAMESPACE for tests
            add_defines("SHORT_NAMESPACE")

            add_syslinks("pthread")

            -- Add as test
            add_tests("default", {rundir = os.projectdir()})
        target_end()
    end
end

-- Task to generate CMakeLists.txt
task("cmake")
    on_run(function ()
        import("core.project.config")

        -- Load configuration
        config.load()

        -- Generate CMakeLists.txt
        os.exec("xmake project -k cmakelists")

        print("CMakeLists.txt generated successfully!")
    end)

    set_menu {
        usage = "xmake cmake",
        description = "Generate CMakeLists.txt from xmake.lua",
        options = {}
    }
task_end()
