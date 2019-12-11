const core = require('@actions/core');
const io = require('@actions/io');
const exec = require('@actions/exec');

async function run() {
    try {
        // `who-to-greet` input defined in action metadata file
        const target = core.getInput('target', { required: true });
        const size = core.getInput('size', { required: true });

        // Generatest test cases
        core.debug("Generating test file, size: " + size + " entries");
        await exec.exec("python", ["gen.py", "test_" + size, size]);

        // Run program
        core.debug("Run program");
        let runOutput = '';
        let runError = '';

        const options = {};
        options.listeners = {
            stdout: (data) => {
                runOutput += data.toString();
            },
            stderr: (data) => {
                runError += data.toString();
            }
        };

        await exec.exec("eval", ["'(time", "./build/fmindex/" + target, "test_" + size + ".txt", size + ")", "2>&1", "|", "awk", String.raw`"/real\t([0-9]+m[0-9]+.[0-9]+s)/"'`], options);

        core.setOutput("runtime", runOutput);
    } catch (error) {
        core.setFailed(error.message);
    }
}

run()