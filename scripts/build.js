const core = require('@actions/core');
const io = require('@actions/io');
const exec = require('@actions/exec');

async function run() {
    try {
        // `who-to-greet` input defined in action metadata file
        const target = core.getInput('target', { required: true });
        const cmake_build_type = core.getInput('cmake_build_type', { required: true });

        core.debug("Start building");
        const build_path = 'build';

        core.debug(":: Create build dir");
        await io.mkdirP(build_path);

        const options = {};
        options.cwd = "./" + build_path;

        core.debug("Run CMake");
        await exec.exec("cmake", ["..", "-DCMAKE_BUILD_TYPE="+cmake_build_type], options);

        core.debug("Run Make");
        await exec.exec("make", [target], options);

    } catch (error) {
        core.setFailed(error.message);
    }
}

run()