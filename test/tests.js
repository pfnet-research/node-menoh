'use strict';

const menoh = require('..');
const jimp = require("jimp");
const fs = require('fs');
const _ = require('lodash');
const assert = require('assert');

const MNIST_IN_NAME  = '139900320569040'
const MNIST_OUT_NAME = '139898462888656'
const INPUT_IMAGE_LIST = [
    './test/data/mnist/0.png',
    './test/data/mnist/1.png',
    './test/data/mnist/2.png',
    './test/data/mnist/3.png',
    './test/data/mnist/4.png',
    './test/data/mnist/5.png',
    './test/data/mnist/6.png',
    './test/data/mnist/7.png',
    './test/data/mnist/8.png',
    './test/data/mnist/9.png'
];
const ONNX_FILE_PATH = './test/data/mnist/mnist.onnx';

function loadInputImages(paths) {
    return Promise.all(paths.map((filename) => jimp.read(filename)));
}

function preprocessImages(imageList) {
    const data = [];
    imageList.forEach((image, batchIdx) => {
        // All the input images are already croped and resized to 28 x 28.
        assert.equal(image.bitmap.width, 28);
        assert.equal(image.bitmap.height, 28);

        // Convert bitmap to an array. (Use R channel only - already in greyscale)
        const numPixels = image.bitmap.width * image.bitmap.height;
        const batchOffset = batchIdx * numPixels;
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
            const dataIdx = y * image.bitmap.width + x + batchOffset;
            data[dataIdx] = this.bitmap.data[idx]; // R channel
        });
    });
    return data;
}

// Find the indexes of the k largest values.
function findIndicesOfTopK(a, k) {
    var outp = [];
    for (var i = 0; i < a.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > k) {
            outp.sort((l, r) => { return a[r] - a[l]; });
            outp.pop();
        }
    }
    return outp;
}

function validateOutput(model, batchSize) {
    const out = model.getOutput(MNIST_OUT_NAME);
    // sanity check
    assert.equal(out.dims[0] * out.dims[1], out.data.length);
    assert.equal(out.dims[0], batchSize);

    // Evaluate results.
    out.data = _.chunk(out.data, out.dims[1]); // reshaped
    for (let bi = 0; bi < batchSize; ++bi) {
        const topK = findIndicesOfTopK(out.data[bi], 1);
        topK.forEach((idx) => {
            assert.equal(idx, bi);
        });
    }
}

describe('MNIST tests', function () {
    let imageList;
    let batchSize;
    let data;

    before(function () {
        return loadInputImages(INPUT_IMAGE_LIST)
        .then((_imageList) => {
            imageList = _imageList;
            batchSize = imageList.length;
            data = preprocessImages(imageList);
        });
    })

    it('Get native version', function () {
        const v = menoh.getNativeVersion();
        assert.equal(typeof v, 'string');
        assert.equal(v.split('.').length, 3);
    });

    it('Succeed with callback', function (done) {
        // Load ONNX file
        menoh.create(ONNX_FILE_PATH, (err, builder) => {
            assert.ifError(err);
            const batchSize = imageList.length;

            builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
            builder.addOutput(MNIST_OUT_NAME);

            // Make a new Model
            const model = builder.buildModel({
                backendName: 'mkldnn'
            })

            model.setInputData(MNIST_IN_NAME, data);

            // Run the model
            model.run((err) => {
                assert.ifError(err);
                validateOutput(model, batchSize);
                done();
            });
        });
    });

    it('Succeed with promise', function () {
        // Load ONNX file
        return menoh.create(ONNX_FILE_PATH)
        .then((builder) => {
            const batchSize = imageList.length;

            builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
            builder.addOutput(MNIST_OUT_NAME);

            // Make a new Model
            const model = builder.buildModel({
                backendName: 'mkldnn'
            })

            model.setInputData(MNIST_IN_NAME, data);

            // Run the model
            return model.run()
            .then(() => {
                validateOutput(model, batchSize);
            });
        });
    });

    it('Run the same model more than once', function () {
        // Load ONNX file
        return menoh.create(ONNX_FILE_PATH)
        .then((builder) => {
            const batchSize = imageList.length;

            builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
            builder.addOutput(MNIST_OUT_NAME);

            // Make a new Model
            const model = builder.buildModel({
                backendName: 'mkldnn'
            })

            model.setInputData(MNIST_IN_NAME, data);

            // Run the model
            return model.run()      // 1st run
            .then(() => {
                validateOutput(model, batchSize);
                return model.run(); // 2nd run
            })
            .then(() => {
                validateOutput(model, batchSize);
                return model.run(); // 3rd run
            })
            .then(() => {
                validateOutput(model, batchSize);
            });
        });
    });

    it('Run two models concurrently', function () {
        // Load ONNX file
        return menoh.create(ONNX_FILE_PATH)
        .then((builder) => {
            const batchSize = imageList.length;

            builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
            builder.addOutput(MNIST_OUT_NAME);

            // create model1
            const model1 = builder.buildModel({
                backendName: 'mkldnn'
            })
            model1.setInputData(MNIST_IN_NAME, data);

            // create model2
            const model2 = builder.buildModel({
                backendName: 'mkldnn'
            })
            model2.setInputData(MNIST_IN_NAME, data);

            // Run these models concurrently
            return Promise.all([ model1.run(), model2.run() ])
            .then(() => {
                [model1, model2].forEach((model) => {
                    validateOutput(model, batchSize);
                });
            });
        });
    });
});

describe('Failure tests with callback', function () {
    let imageList;
    let batchSize;
    let data;

    before(function () {
        return loadInputImages(INPUT_IMAGE_LIST)
        .then((_imageList) => {
            imageList = _imageList;
            batchSize = imageList.length;
            data = preprocessImages(imageList);
        });
    })

    describe('#create tests', function () {
        it('should throw when the path value is invalid', function () {
            assert.throws(() => {
                menoh.create(20, function () {});
            }, (err) => {
                if (err instanceof TypeError) {
                    return true;
                }
                if (err.message.includes('arg 1')) {
                    return true;
                }
            }, 'unexpected error');
        });

        it('should throw when the path does not exist', function (done) {
            menoh.create('bad_onnx_file_path', function (err) {
                assert.ok(err);
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('bad_onnx_file_path'));
                done();
            });
        });
    });

    describe('#buildModel tests', function () {
        it('should throw with invalid input name', function (done) {
            menoh.create(ONNX_FILE_PATH, function (err, builder) {
                assert.ifError(err)

                builder.addInput('bad_input_name', [ batchSize, 1, 28, 28 ]);

                assert.throws(() => {
                    const model = builder.buildModel({
                        backendName: 'mkldnn'
                    });
                }, (err) => {
                    assert.ok(err instanceof Error);
                    assert.ok(err.message.includes('variable not found'));
                    return true;
                });
                done();
            });
        });

        it('should throw with invalid output name', function (done) {
            menoh.create(ONNX_FILE_PATH, function (err, builder) {
                assert.ifError(err)

                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput('bad_output_name');

                assert.throws(() => {
                    const model = builder.buildModel({
                        backendName: 'mkldnn'
                    });
                }, (err) => {
                    assert.ok(err instanceof Error);
                    assert.ok(err.message.includes('variable not found'));
                    return true;
                });
                done();
            });
        });
    });
});

describe('Failure tests with promise', function () {
    let imageList;
    let batchSize;
    let data;

    before(function () {
        return loadInputImages(INPUT_IMAGE_LIST)
        .then((_imageList) => {
            imageList = _imageList;
            batchSize = imageList.length;
            data = preprocessImages(imageList);
        });
    })

    describe('#create tests', function () {
        it('should fail when the path value is invalid', function () {
            return menoh.create(20)
            .then(assert.fail, (err) => {
                assert.ok(err instanceof TypeError);
                assert.ok(err.message.includes('arg 1'));
            });
        });

        it('should fail when the path does not exist', function () {
            return menoh.create('bad_onnx_file_path')
            .then(assert.fail, (err) => {
                assert.ok(err);
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('bad_onnx_file_path'));
                return true;
            });
        });
    });

    describe('#addInput tests', function () {
        it('should throw with insufficient number of args', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput('bad_input_name')
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('insufficient'));
            });
        });
        it('should throw with invalid arg 1', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(666, [ batchSize, 1, 28, 28 ]);
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('arg 1'));
            });
        });
        it('should throw with invalid arg 2', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, 'bad_arg_2');
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('arg 2'));
            });
        });
        it('should throw with invalid dims size', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize ]);
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('dims'));
            });
        });
    });

    describe('#addOutput tests', function () {
        it('should throw with insufficient number of args', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput();
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('insufficient'));
            });
        });
        it('should throw with invalid arg 1', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(666);
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('arg 1'));
            });
        });
    });

    describe('#buildModel tests', function () {
        it('should throw with invalid input name', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput('bad_input_name', [ batchSize, 1, 28, 28 ]);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('variable not found'));
            });
        });

        it('should throw with invalid output name', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput('bad_output_name');
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('variable not found'));
            });
        });
    });

    describe('#setInputData tests', function () {
        it('should throw with invalid input data', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                model.setInputData('bad_input_name', data); // should throw
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('variable not found'));
            });
        });

        it('should throw if input data is too short', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                const tooShort = [0, 1, 2];
                model.setInputData(MNIST_IN_NAME, tooShort); // should throw
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('too short'));
            });
        });

        it('should throw if input data is too long', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                const tooLong = data.concat([0.666]);
                model.setInputData(MNIST_IN_NAME, tooLong); // should throw
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('too long'));
            });
        });
    });

    describe('#run tests', function () {
        it('should throw with invalid arg 1', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                model.setInputData(MNIST_IN_NAME, data);

                model.run('bad')
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('arg 1'));
            });
        });
        it('second run() on the same model should fail', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                model.setInputData(MNIST_IN_NAME, data);

                let err1 = null;
                let err2 = null;

                return Promise.all([
                    model.run().catch((err) => {
                        err1 = err;
                    }),
                    model.run().catch((err) => {
                        err2 = err;
                    }),
                ])
                .then(() => {
                    assert.ok(!err1);
                    assert.ok(err2 instanceof Error);
                    assert.ok(err2.message.includes('in progress'));

                    validateOutput(model, batchSize);
                });
            })
        });
    });

    describe('#getOutput tests', function () {
        it('should throw with invalid output name', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                model.setInputData(MNIST_IN_NAME, data);

                return model.run()
                .then(() => {
                    model.getOutput(); // should throw
                })
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('insufficient'));
            });
        });
        it('should throw with invalid type of output name', function () {
            return menoh.create(ONNX_FILE_PATH)
            .then((builder) => {
                builder.addInput(MNIST_IN_NAME, [ batchSize, 1, 28, 28 ]);
                builder.addOutput(MNIST_OUT_NAME);
                const model = builder.buildModel({
                    backendName: 'mkldnn'
                });

                model.setInputData(MNIST_IN_NAME, data);

                return model.run()
                .then(() => {
                    model.getOutput(666); // should throw
                })
            })
            .then(assert.fail, (err) => {
                assert.ok(err instanceof Error);
                assert.ok(err.message.includes('arg 1'));
            });
        });
    });
});
