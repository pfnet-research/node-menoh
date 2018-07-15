'use strict';

const jimp = require("jimp");
const fs = require('fs');
const ndarray = require('ndarray');
const dtype = require('dtype');
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
    for (var i = 0; i < a.size; i++) {
        outp.push(i); // add index to output array
        if (outp.length > k) {
            outp.sort((l, r) => { return a.get(r) - a.get(l); });
            outp.pop();
        }
    }
    return outp;
}

console.log('Using menoh core version %s', menoh.getNativeVersion());

const categoryList = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

/*
*/

// Load ONNX file
return menoh.create('../test/data/mnist/mnist.onnx')
.then((builder) => {
    const batchSize = INPUT_IMAGE_LIST.length;

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

    // Create a view for input buffer using ndarray.
    const iData = (function () {
        const prof = model.getProfile(MNIST_IN_NAME);
        return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
    })();

    // Create a view for output buffer using ndarray.
    const oData = (function () {
        const prof = model.getProfile(MNIST_OUT_NAME);
        return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
    })();

    return loadInputImages()
    .then((imageList) => {
        imageList.forEach((image, batchIdx) => {
            // All the input images are already croped and resized to 28 x 28.
            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                const val = image.bitmap.data[idx];
                iData.set(batchIdx, 0, y, x, val);
            });
        });

        // Run the model
        return model.run()
        .then(() => {
            // Print the results.
            for (let bi = 0; bi < batchSize; ++bi) {
                console.log('### Result for %s', INPUT_IMAGE_LIST[bi]);

                const topK = findIndicesOfTopK(oData.pick(bi, null), 1);
                topK.forEach((i) => {
                    console.log('[%d] %f %s', i, oData.get(bi, i), categoryList[i]);
                });
            }
        });
    });
})
.catch((err) => {
    console.log('Error:', err);
});

