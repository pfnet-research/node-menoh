'use strict';

const jimp = require("jimp");
const assert = require('assert');
const fs = require('fs');
const _ = require('lodash');
const menoh = require('..'); // This menoh module

const MNIST_IN_NAME = "139900320569040"
const MNIST_OUT_NAME = "139898462888656"
const INPUT_IMAGE_LIST = [
    "../test/data/mnist/0.png",
    "../test/data/mnist/1.png",
    "../test/data/mnist/2.png",
    "../test/data/mnist/3.png",
    "../test/data/mnist/4.png",
    "../test/data/mnist/5.png",
    "../test/data/mnist/6.png",
    "../test/data/mnist/7.png",
    "../test/data/mnist/8.png",
    "../test/data/mnist/9.png"
];

// Load all image files.
function loadInputImages() {
    return Promise.all(INPUT_IMAGE_LIST.map((filename) => jimp.read(filename)));
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

console.log('Using menoh core version %s', menoh.getNativeVersion());

const categoryList = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

loadInputImages()
.then((imageList) => {
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

    // Load ONNX file
    return menoh.create('../test/data/mnist/mnist.onnx')
    .then((builder) => {
        const batchSize = imageList.length;

        // Add input data
        builder.addInput(MNIST_IN_NAME, [
            batchSize,  // 10 images in the data
            1,          // number of channels
            28,         // height
            28          // width
        ]);

        // Add output
        builder.addOutput(MNIST_OUT_NAME);

        // Build a new Model
        const model = builder.buildModel({
            backendName: 'mkldnn'
        })

        // Set input data
        model.setInputData(MNIST_IN_NAME, data);

        // Run the model
        return model.run()
        .then(() => {
            const out = model.getOutput(MNIST_OUT_NAME);

            // just to be sure
            assert.equal(out.dims[0] * out.dims[1], out.data.length);
            assert.equal(out.dims[0], batchSize); // only applies to this example

            // Print the results.
            out.data = _.chunk(out.data, out.dims[1]); // reshaped
            
            for (let bi = 0; bi < batchSize; ++bi) {
                console.log('### Result for %s', INPUT_IMAGE_LIST[bi]);

                const topK = findIndicesOfTopK(out.data[bi], 1);
                topK.forEach((i) => {
                    console.log('[%d] %f %s', i, out.data[bi][i], categoryList[i]);
                });
            }

            // Happily done!
        });
    });
});

